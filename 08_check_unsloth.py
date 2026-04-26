"""
Unslothが正しくインストールされているか、GPUが認識されているかを確認するためのスクリプト。
このスクリプトを実行して、GPU名とVRAM容量が正しく表示されれば、Unslothの環境は正常にセットアップされている。
"""
import torch
from unsloth import FastLanguageModel

# GPU名の確認
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# VRAM確認（プロパティから直接取得）
props = torch.cuda.get_device_properties(0)
print(f"VRAM Total (Properties): {props.total_memory / 1024**3:.2f} GB")

print("Unsloth is ready!")
