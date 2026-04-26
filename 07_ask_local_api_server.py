"""
07_ask_local_api_server.py
ローカル環境にダウンロードしたLLMを、APIサーバーとして立てて、OpenAI互換のAPI経由で呼び出すサンプルコード。
APIサーバーの実装には、llama-cpp-python の LlamaServer クラスを使用。
クライアント側は、OpenAIのPython SDKを使って、ローカルAPIサーバーにリクエストを送るコード例を示す。
サーバーの起動方法や、APIのエンドポイント、リクエストのフォーマットなども解説。
※ 事前に、05_run_api_server.bash を実行してAPIサーバーを起動しておいてください。
"""
from typing import Generator
from openai import OpenAI

# サーバーのURL（05_run_api_server.bash で設定したポート）を指定
client = OpenAI(base_url="http://localhost:8080/v1", api_key="llamacpp")

def ask_local_api_server(prompt:str) -> Generator[str, None, None]:
    # ストリーム生成
    response = client.chat.completions.create(
        model="local-model", # サーバー側で読み込んでいるため、この文字列はなんでもよい
        messages=[{ "role": "user", "content": prompt }],
        stream=True,
        temperature=0.7,
        max_tokens=512
    )

    # 応答に対し、逐次表示
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content

if __name__ == "__main__":
    # 1. 質問の準備
    prompt = "兵庫県姫路市でおすすめの観光スポットを3つ教えてください。"
    # prompt = "兵庫県姫路市にあるお城の名前は何ですか？"

    # 2. APIサーバーにリクエストを送って、応答を逐次表示
    results = ask_local_api_server(prompt)
    print("----- response -----")
    for text in results:
        print(text, end="", flush=True)
    print("\n--------------------")
