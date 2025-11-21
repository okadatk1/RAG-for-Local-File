# RAG-for-Local-File
# RAG Index API (Hugging face)

ローカルでRAG（Retrieval-Augmented Generation）を構築し、LLMに問い合わせるAPI。  
ファイルをネットワークに送信せずにローカルでIndex作成が可能。  

---

## 🔧 概要

- **Index作成**: 指定フォルダのテキストファイル・PDF・DOCXをVectorStoreIndex化
- **Query**: 作成したIndexを使ってLLMに質問
- **Embedding**: Huggingface Localでは小さいことが必須
- **LLM**: Ollama
- **Streamlit**でREST API提供

---- 完全ローカル運用（ファイルは外に出さない）を最優先。Embedding と LLM を Ollama に固定。
- Index 作成は軽量で高速な embedding（`nomic-embed-text` 等）を使う。
- Query（生成）は軽量な LLM（例: `llama3.2:3b`）を推奨。大きいモデルはタイムアウト／OOMの要因。
- `load_index_from_storage(..., embed_model=...)` を使ってロード時にも embedding を明示的に渡す。
- Index のメタ（`meta.json`）を保存し、一貫性チェックに用いる。
- `request_timeout` は初回ロード考慮で十分大きく（例: 180s）。

### ■ インデックス化
- 指定したフォルダーを再帰的に走査し全文を抽出 PDF, DOCX, XLSX, PPTX, CSV, TXT, MD, ZIP（解凍して再帰読み込み）
- 進捗（ファイル読み込みフェーズ & インデックス生成フェーズ）を Streamlit 上で可視化
- chunk_size / chunk_overlap 調整可能
- HuggingFace 埋め込みモデル（ローカル）を使用
- インデックスは自動的に削除して再構築
- meta.json に埋め込みモデル情報を記録し、クエリ時にモデル不一致があればエラー
### ■ 検索
- 作成済みインデックス一覧を選択式で表示
- Ollama のモデル一覧を GUI 上で選択
- 選んだモデルで LLM 生成（LLM を使わず Retriever のみモードも可能に拡張可）
- 完全ローカルで動作（外部送信なし）
### ■ UI仕様（Streamlit）
- 左: Embeddingモデルパス、Ollamaモデル、インデックス一覧
- 「インデックス作成」タブ → フォルダー/インデックス名/パラメータ/キャンセルボタン
- 「検索」タブ → LLMモデル + クエリ + 実行ボタン

## 2) 前提 & 環境
- OS: WSL2 on Windows（他のLinuxでも同等）
- Python 3.11+ 推奨

## 📁 ディレクトリ構成
rag_project/
├─ app.py # UI
├─ rag_core.py # Index作成、検索本体
├─ requirements.txt # Python依存関係
├─ .env # 環境変数
├─ storage/ # 作成したIndexを格納
└─ README.md


## ⚙️ 環境変数 (.env)

```dotenv
STORAGE_DIR=storage
LLM_PROVIDER=ollama  # デフォルトプロバイダ

# Ollama
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2

# OpenAI (必要なら)
OPENAI_API_KEY=your_openai_key
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o
  ```

## 📦 必要ライブラリ (requirements.txt)
```requirements.txt
streamlit>=1.18.0
llama-index==0.9.44
ollama>=0.1.0
transformers>=4.30.0
sentence-transformers>=2.2.2
pdfminer.six
python-docx
docx2txt
openpyxl
chardet
tqdm
  ```

注意: 最新の llama_index と ollama を推奨。OpenAI呼び出しも可能。

## 🚀 実行方法
### 環境構築
'''
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
  ```
### API起動
'''
uvicorn main:app --reload
  ```

### APIエンドポイント:

POST /index/build - フォルダ指定でIndex作成
POST /index/query - 作成済Indexに問い合わせ
GET /health - ヘルスチェック

## 📝 API使用例
Index作成
'''
POST /index/build
{
  "folder_path": "data/docs",
  "index_name": "TestFolder_index",
  "provider": "ollama"
}
  ```
'''
Query
POST /index/query
{
  "index_name": "TestFolder_index",
  "query": "この文書の要点は？",
  "use_llm_provider": "ollama"
}
  ```
## ⚠️ Troubleshooting & Tips

1. Embedding / LLM のプロバイダ不一致

作成時とQuery時でプロバイダが異なるとエラー
Index embedding <class 'llama_index.embeddings.openai.base.OpenAIEmbedding'> does not match requested provider ollama.
対策: インデックス作成とQueryで同じプロバイダを使用。必要に応じて古いIndexを削除。

2. Ollama LLM タイムアウト

大きなモデルではQuery時にタイムアウト発生
httpcore.ReadTimeout: timed out
対策: request_timeout=180 など長めに設定。軽量モデルでIndex作成。

3. Index作成時の古い残骸
Indexフォルダに古いファイルが残っている場合、Embeddingが変わるとエラー
対策: shutil.rmtree(index_path) で削除してから作成

4. ファイル読み込みのエラー
.txt: OK
.pdf: PyPDFがサポートしない場合あり
ERROR:pypdf._cmap:Advanced encoding /90ms-RKSJ-V not implemented yet
.docx: docx2txt 必須
pip install docx2txt

読み込めないファイルは skipped_files に記録

5. OpenAI API Key 認証エラー
.env に正しいキーを設定しないと認証エラー
Incorrect API key provided

対策: .env に OPENAI_API_KEY を設定。Gitに入れない。
6. Index JSON / SQLite
Indexはローカル保存（JSON / StorageContext）でOK
SQLite必須ではない
_embed_model でEmbedding情報確認可能

7. モデルサイズとローカル実行

Index作成は小さいモデルで十分（例：llama3.2:3b）
Query時に大きいモデルで精度向上可能
8. その他Tips

.env と storage/ はGit管理しない
複数フォルダ対応は safe_load_documents_from_folder の拡張で対応可能
Query精度向上にはドメイン別Index、Embedモデル・LLMモデルの選択を適切に
