"""
04_ask_local_by_llamacpp.py
ローカル環境にダウンロードしたLLMを、llama-cpp-python の Llama クラスを使ってテキスト生成するサンプルコード。
※ サーバーとして立ててAPI経由で呼び出す方法は、07_ask_local_api_server.py を参照してください。
Llama-3.1-8B-Instruct-Q6_K.gguf を選択した理由
  - FP16（無量子化）とほぼ遜色ない精度を保ちつつ、ファイルサイズを大幅に削減できる
  - VRAM消費: 約6.6GB前後。RTX 5070の8GBにOSの消費分を含めても、「全レイヤーGPUロード（n_gpu_layers=-1）」で最速動作が可能。
"""
from typing import Generator
from llama_cpp import Llama

model_path = r"/home/ct1485/_local_wsl/08_tools/models/gguf/Llama-3.1-8B-Instruct-Q6_K.gguf"
# model_path = r"/home/ct1485/_local_wsl/08_tools/models/gguf/Llama-3.3-8B-Instruct.Q4_K_S.gguf"
# model_path = r"/home/ct1485/_local_wsl/02_study/20260222_llm/model_gozaru_gguf/model.gguf" # ファインチューニングしたござるモデル


def ask_local_llamacpp(llm, prompt) -> Generator[str, None, None]:
    """ローカルのLlamaモデルにプロンプトを投げて、応答を逐次的に取得するジェネレーター関数
    Args:
        llm: llama_cpp の Llama クラスのインスタンス
        prompt: ユーザーからの質問や指示を含む文字列
        
    Returns:
        生成されたテキストを逐次的に返すジェネレーター
    """
    # メッセージの設定
    messages = [
        { "role": "user", "content": prompt }
    ]

    # llama-cpp-python の Llama クラスの create_chat_completion メソッドを呼び出して、ストリーミングで応答を取得
    response_stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        temperature=0.7, # 創造性の調整
        repeat_penalty=1.2, # 繰り返しのペナルティ
        max_tokens=512   # 最大出力トークン数
    )

    # 逐次取得して表示
    for chunk in response_stream:
        if "content" in chunk["choices"][0]["delta"]:
            content = chunk["choices"][0]["delta"]["content"]
            yield content

if __name__ == "__main__":
    # 1. モデルのロード
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1, # 全てのレイヤーをGPUに転送する（VRAM不足なら数字を減らす）
        n_ctx=2048,      # 文脈サイズ（必要に応じて調整）
        verbose=False    # 起動時のログを非表示にする
    )

    # 2. 質問の準備
    prompt = "兵庫県姫路市でおすすめの観光スポットを3つ教えてください。"
    # prompt = "兵庫県姫路市にあるお城の名前は何ですか？"

    # 3. APIサーバーにリクエストを送って、応答を逐次表示
    results = ask_local_llamacpp(llm, prompt)
    print("----- response -----")
    for text in results:
        print(text, end="", flush=True)
    print("\n--------------------")