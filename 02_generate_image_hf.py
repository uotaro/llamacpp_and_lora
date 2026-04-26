"""
02_generate_image_hf.py
Hugging FaceのInference APIを使って、テキストから画像を生成するサンプルコード。1か月あたりの使用制限あり
Hugging Faceの API_KEY を取得する必要がある。
当コードは、取得した API_KEY を環境変数 HF_TOKEN としてる前提で作成。
"""
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="together",
    api_key=os.environ["HF_TOKEN"],
)

# request_text = "A fantasy forest with glowing mushrooms"
request_text = "A fantasy forest with a beautiful golden deer"

# output is a PIL.Image object
image = client.text_to_image(
    request_text,
    model="black-forest-labs/FLUX.1-dev",
)

# ローカル環境のビューアーで画像を表示
image.show()

# ファイルとして保存
# image.save("output_image.png")
