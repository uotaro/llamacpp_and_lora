
"""
10_convert_to_gguf.py
09_train_gozaru_ROLA.py で保存した LoRAアダプターをベースモデルにマージして、gguf形式に変換するスクリプト。
このスクリプトを実行すると、以下の工程が自動で行われます。
    1. ベースモデルを16bitで直接ロード（4bitのロードは完全回避）
    2. LoRAアダプターをマージして、全テンソルをbfloat16に統一
    3. HF Transformers形式で保存
    4. llama.cpp の convert_hf_to_gguf.py を呼び出して、gguf形式に変換
最終的に、05_run_api_server.bash で使用可能な gguf ファイルが生成されます。
実行前に、09_train_gozaru_ROLA.py を実行して、LoRAアダプターが保存されていることを確認してください。
また、llama.cpp の convert_hf_to_gguf.py が正しいパスに存在していることも確認してください。
生成された gguf ファイルは、llamacpp-server で使用することができます。
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unsloth import FastLanguageModel
import os
import subprocess

# ========== パス設定 ==========
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "/home/ct1485/_local_wsl/08_tools/models/unsloth/Qwen2.5-3B-Instruct"
LORA_PATH   = os.path.join(SCRIPT_DIR, "model_gozaru_lora")
MERGED_PATH = os.path.join(SCRIPT_DIR, "model_gozaru_merged")
GGUF_DIR    = os.path.join(SCRIPT_DIR, "model_gozaru_gguf")
GGUF_PATH   = os.path.join(GGUF_DIR, "model.gguf")
# LLAMA_CPP   = "/home/ct1485/.unsloth/llama.cpp"
LLAMA_CPP   = "/home/ct1485/_local_wsl/08_tools/llama.cpp"

# ========== 事前チェック ==========
print("=== 事前チェック ===")

if not os.path.isdir(LORA_PATH):
    raise FileNotFoundError(
        f"LoRAアダプターが見つかりません: {LORA_PATH}\n"
        f"先に 09_train_gozaru_ROLA.py を実行してください。"
    )
print(f"✅ LoRAアダプター確認: {LORA_PATH}")

converter = os.path.join(LLAMA_CPP, "convert_hf_to_gguf.py")
if not os.path.exists(converter):
    raise FileNotFoundError(
        f"convert_hf_to_gguf.py が見つかりません: {converter}"
    )
print(f"✅ llama.cpp変換スクリプト確認: {converter}")

# ========== Step1: ベースモデルを16bitで直接ロード ==========
print("[1/3] ベースモデルを16bitで直接ロード中...")

import torch
from peft import PeftModel

# Unslothではなく通常のtransformersで読む（4bit完全回避）
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype = torch.bfloat16,  # 明示的にbfloat16
    device_map  = "cuda",
)
print("✅ ベースモデルロード完了")

# LoRAをマージ
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()
print("✅ LoRAマージ完了")

# 全テンソルをbfloat16に統一
model = model.to(torch.bfloat16)
print("✅ bfloat16に変換完了")

# ========== Step2: HF形式で保存 ==========
print(f"\n[2/3] HF形式で保存中...")
os.makedirs(MERGED_PATH, exist_ok=True)
model.save_pretrained(MERGED_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_PATH)

if not os.path.isfile(os.path.join(MERGED_PATH, "config.json")):
    raise FileNotFoundError(f"保存失敗: {MERGED_PATH}")
print("✅ 保存完了")

# ========== Step3: GGUF変換 ==========
print(f"\n[3/3] GGUF変換中（q8_0）...")
print(f"      出力先: {GGUF_PATH}")
os.makedirs(GGUF_DIR, exist_ok=True)

cmd = [
    "python3",
    converter,
    MERGED_PATH,
    "--outfile", GGUF_PATH,
    "--outtype", "q8_0",
]
print(f"      実行コマンド: {' '.join(cmd)}")
subprocess.run(cmd, check=True)

# ========== 完了 ==========
print(f"""
✅ 全工程完了！

📁 出力ファイル: {GGUF_PATH}

🚀 llama-serverで使う場合:
   SERVER_EXE=/home/ct1485/_local_wsl/08_tools/llama.cpp/build/bin/llama-server
   MODEL_PATH={GGUF_PATH}
   
   bash 05_run_api_server.bash  # SERVER_EXEとMODEL_PATHを上記に変更してから
""")