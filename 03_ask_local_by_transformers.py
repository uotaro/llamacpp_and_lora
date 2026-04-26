"""
03_ask_local_by_transformers.py
ローカル環境にダウンロードしてLLMを、transformers の generate を直接呼び出してテキスト生成するサンプルコード。
Unslothのバグを回避するため、generate の呼び出し方に工夫が必要。
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_DIR = "/home/ct1485/_local_wsl/08_tools/models/unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

SYSTEM_PROMPT = "あなたは親切で正確な日本語アシスタントです。"
USER_PROMPT = "兵庫県姫路市でおすすめの観光スポットを5つ教えてください。"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.2


def main():
    print("[INFO] Loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading model...", file=sys.stderr)

    # GPUが使えない/不安定な場合でもまず動かしやすいように auto
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    # Unslothのバグ回避のため、generate を別スレッドで呼び出す工夫が必要
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True if TEMPERATURE > 0 else False,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("----- response -----")
    for text in streamer:
        print(text, end="", flush=True)

    thread.join()
    print("\n--------------------")


if __name__ == "__main__":
    main()