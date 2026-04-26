"""
01_generate_hf.py
Hugging FaceのAPIを使って、簡単なチャット形式のテキスト生成を試すコード。1か月あたりの使用制限あり
Hugging Faceの API_KEY を取得する必要がある。
当コードは、取得した API_KEY を環境変数 HF_TOKEN としてる前提で作成。
"""
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[
        {
            "role": "user",
            "content": "兵庫県姫路市のおすすめの観光スポットを5つ教えてください。"
        }
    ],
)

message = completion.choices[0].message
content = message.content
print(content)