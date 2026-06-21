# llamacpp_and_lora
Ubuntuローカルでのllamacppサーバー構築方法とLORA練習

### 01_generate_hf.py
Hugging FaceのAPIを使って、簡単なチャット形式のテキスト生成を試すコード。  
Hugging FaceのAPIは、1か月あたりの使用制限あり。利用するには Hugging Faceの API_KEY を取得する必要がある。

### 02_generate_image_hf.py
Hugging FaceのInference APIを使って、テキストから画像を生成するサンプルコード。  
Hugging FaceのAPIは、1か月あたりの使用制限あり。利用するには Hugging Faceの API_KEY を取得する必要がある。

### 03_ask_local_by_transformers.py
ローカル環境にダウンロードしてLLMを、transformers の generate を直接呼び出してテキスト生成。  
Unslothのバグを回避するため、generate の呼び出し方に工夫が必要。

### 04_ask_local_by_llamacpp.py
ローカル環境にダウンロードしたLLMを、llama-cpp-python の Llama クラスを使ってテキスト生成。  
※ サーバーとして立ててAPI経由で呼び出す方法は、07_ask_local_api_server.py を参照

### 05_run_api_server.bash
Llama-cpp API Server 起動用スクリプト。  
実行前に 3項目設定が必要。当 bash の冒頭コメント参照

### 06_stop_api_server.bash
Llama-cpp API Server 停止用スクリプト

### 07_ask_local_api_server.py
`05_run_api_server.bash` で起動させた Llama-cpp API サーバーに対し、OpenAI互換のAPI経由で呼び出す。

### 08_check_unsloth.py
Unslothが正しくインストールされているか、GPUが認識されているかを確認するためのスクリプト。

### 09_train_gozaru_ROLA.py
UnslothのFastLanguageModelを使って、ござるデータセットに対してLoRA（QLoRA）でファインチューニングを行う。  
学習後、モデルの推論テストも行い、最後にLoRAアダプターを保存するところまでを1ファイルで完結

### 10_convert_to_gguf.py
`09_train_gozaru_ROLA.py` で保存した LoRAアダプターをベースモデルにマージして、gguf形式に変換するスクリプト。
当スクリプトを実行すると、最終的に `05_run_api_server.bash` で使用可能な gguf ファイルが生成される。
