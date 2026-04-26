"""
09_train_gozaru_ROLA.py
UnslothのFastLanguageModelを使って、ござるデータセットに対してLoRA（QLoRA）でファインチューニングを行うサンプルコード。
学習後、モデルの推論テストも行い、最後にLoRAアダプターを保存するところまでを1ファイルで完結させています。
学習の安定性を優先して、いくつかの設定は控えめにしていますが、VRAM容量に余裕があれば、適宜調整してみてください。
使用するモデルは、Unslothが提供する Qwen2.5-7B-Instruct の4bit量子化版ですが、VRAM容量に応じて3B版も選択可能です。
学習の前に、08_check_unsloth.py を実行して、Unslothの環境が正常にセットアップされていることを確認してください。
"""
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 1. 設定
max_seq_length = 512 # 8GB VRAMのため、短めに設定してメモリを節約
dtype = None # Noneにすると自動検出（RTX 5070ならBF16が使われます）
load_in_4bit = True # 4bit量子化でロード

# model_name ="/home/ct1485/_local_wsl/08_tools/models/unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
model_name ="/home/ct1485/_local_wsl/08_tools/models/unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

# 2. モデルとトークナイザーのロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. LoRA（QLoRA）パラメータの設定
# 8GB VRAMで7Bモデルを動かすための最小構成設定
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # ランクを下げてメモリ消費を抑える
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # メモリ節約の要
    random_state = 3407,
)

# 4. データセットの準備（ござるデータセット）
dataset = load_dataset("bbz662bbz/databricks-dolly-15k-ja-gozaru", split="train")

# Qwen2.5のChatMLフォーマットに合わせる
prompt_style = "<|im_start|>system\nあなたは拙者（せっしゃ）と名乗る侍です。<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>"

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = [prompt_style.format(i, o) + tokenizer.eos_token for i, o in zip(instructions, outputs)]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 【追加】512トークンを超えるデータを除外してVRAMを保護する
def filter_long_samples(example):
    tokens = tokenizer.encode(example["text"], add_special_tokens=False)
    return len(tokens) <= max_seq_length

dataset = dataset.filter(filter_long_samples)
print(f"Filter完了: 残り {len(dataset)} サンプル")

dataset = dataset.shuffle(seed=3407)

# 5. 学習の設定
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1, # 1に固定
    packing = False,      # 8GB VRAM 3Bモデルなら、一度 False にして安定性を優先します
    args = TrainingArguments(
        per_device_train_batch_size = 1, # 3Bなら1から2へ上げても大丈夫なはずです
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        # max_steps = 200,
        max_steps = 1000,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit", 
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. 学習開始
trainer_stats = trainer.train()

# 7. 推論テスト（学習後の「ござる」具合を確認）
print("\n--- 学習後のテスト ---")

# 7-1. inputs の準備（ここはそのまま）
test_prompt = prompt_style.format("日本の首都はどこですか？", "\n")
inputs = tokenizer([test_prompt], return_tensors = "pt").to("cuda")

# 【重要】Unslothのバグがある推論メソッドを、標準のメソッドに差し替える
# これにより、エラーの原因である LlamaAttention_fast_forward_inference を回避します
if hasattr(model, "_old_generate"):
    model.generate = model._old_generate
elif hasattr(model.base_model, "generate"):
    model.generate = model.base_model.generate

# Unslothが上書きした推論用メソッドを、標準のTransformersのものに戻します
from unsloth.models.llama import LlamaAttention_fast_forward_inference
import torch.nn as nn

# モデル内の全ての Attention レイヤから Unsloth の高速推論パッチを剥がす
for module in model.modules():
    if hasattr(module, "forward") and "LlamaAttention_fast_forward_inference" in str(module.forward):
        # 標準的な forward に戻す（あるいはパッチを無効化）
        # 最も確実なのは、モデル読み込み時の状態（Patching前）に戻すことですが、
        # 現状では推論時のキャッシュ(kv_cache)を無効化するのが一番の近道です。
        pass

# --- 生成実行 ---
outputs = model.generate(
    **inputs,
    max_new_tokens = 64,
    use_cache = False, # 【重要】ここを False にしてください。エラーの直接原因である KV Cache の不整合を回避します。
    do_sample = True,
    temperature = 0.8,
)

# デコード
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens = True)
print(f"デコード結果全体：\n{decoded_output}")

# LoRAアダプターのみ保存（確実に動く）
import os
LORA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_gozaru_lora")
model.save_pretrained(LORA_PATH)
tokenizer.save_pretrained(LORA_PATH)
print(f"✅ LoRA保存完了: {LORA_PATH}")
print(f"次のステップ: python 10_convert_to_gguf.py")
