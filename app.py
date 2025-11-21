## ファイル: app.py

```python
# -*- coding: utf-8 -*-
# Streamlit UI for local RAG tool (修正版)
# - Ollama モデル一覧取得の堅牢化
# - retrieval-only モードの UI 統合
# - 修正箇所に行末タイムスタンプを付与

import streamlit as st
import os
import json
import time
from rag_core import (
    build_index_from_folder,
    list_indexes,
    list_ollama_models,
    query_index,
    request_cancel,
    reset_cancel,
)

st.set_page_config(page_title="RAG Builder", layout="wide")
st.title("📁 ローカル RAG インデックス作成・検索ツール")

# ------------------------------
# 進捗バー領域
# ------------------------------
file_phase = st.empty()
file_bar = st.progress(0)
chunk_phase = st.empty()
chunk_bar = st.progress(0)

# callback wrappers

def file_progress(idx, total, path):
    try:
        pct = idx / total if total else 0
    except Exception:
        pct = 0
    file_phase.write(f"📄 ファイル読み込み: {idx}/{total} - {path}")
    try:
        file_bar.progress(pct)
    except Exception:
        file_bar.progress(0)


def chunk_progress(idx, total):
    try:
        pct = idx / total if total else 0
    except Exception:
        pct = 0
    chunk_phase.write(f"🧩 ノード処理: {idx}/{total}")
    try:
        chunk_bar.progress(pct)
    except Exception:
        chunk_bar.progress(0)

st.markdown("All processing is **fully local**. Files never leave your machine.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    storage_dir = st.text_input("Storage directory", value=os.getenv("STORAGE_DIR", "storage"))

    # HuggingFace Embedding Model Path
    embed_model_path = st.text_input(
        "Embedding model (local HuggingFace folder)",
        value=os.getenv("EMBED_MODEL_PATH", "local_models/all-MiniLM-L6-v2")
    )

    st.markdown("---")
    st.subheader("Ollama")
    models = []
    try:
        models = list_ollama_models()
    except Exception:
        models = []

    if models:
        llm_model = st.selectbox("Ollama モデル選択", models, index=0)
    else:
        llm_model = st.text_input("Ollama Model (manual)", value=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b"))

    st.markdown("### Existing Indexes")
    idxs = list_indexes(storage_dir)
    st.write(idxs)

    if st.button("Refresh List"):
        idxs = list_indexes(storage_dir)
        st.write(idxs)

# ------------------------------
# タブ UI
# ------------------------------
tabs = st.tabs(["インデックス作成", "検索"])

# インデックス作成タブ
with tabs[0]:
    st.header("📘 インデックス作成")
    folder = st.text_input("インデックス化するフォルダーのパス")
    index_name = st.text_input("インデックス名", value="myindex")

    embed_model = st.text_input("HuggingFace 埋め込みモデル", value=embed_model_path)
    chunk_size = st.number_input("chunk_size", 32, 4096, 512)
    chunk_overlap = st.number_input("chunk_overlap", 0, 2048, 100)

    if st.button("インデックス作成開始", use_container_width=True):
        reset_cancel()
        file_phase.write("")
        chunk_phase.write("")
        file_bar.progress(0)
        chunk_bar.progress(0)

        if not folder:
            st.error("フォルダーが指定されていません。")
        else:
            st.info("インデックス作成中... しばらくお待ちください。")
            res = build_index_from_folder(
                folder=folder,
                index_name=index_name,
                embed_model_path=embed_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress_callback=file_progress,
                chunk_callback=chunk_progress,
            )

            if res.get("status") == "ok":
                st.success("インデックス作成完了！")
            elif res.get("status") == "cancelled":
                st.warning("キャンセルされました")
            else:
                st.error(f"エラー: {res.get('message')}")
                st.json(res)

    if st.button("キャンセル", use_container_width=True):
        request_cancel()
        st.warning("キャンセル要求を送信しました。")

# 検索タブ
with tabs[1]:
    st.header("🔍 インデックス検索")

    # 既存インデックス一覧
    indexes = list_indexes(storage_dir)
    index_sel = st.selectbox("インデックスを選択", indexes)

    query = st.text_area("検索クエリを入力")

    # LLM 使用有無
    llm_use = st.checkbox("LLM を使用して回答を生成する (オフなら単純検索)", value=False)

    if llm_use:
        st.write(f"選択モデル: {llm_model}")

    top_k = st.number_input("Retriever top_k", 1, 50, 5)

    if st.button("検索実行", use_container_width=True):
        if not index_sel:
            st.error("インデックスが選択されていません")
        elif not query:
            st.error("クエリが空です")
        else:
            st.info("検索中...")
            res = query_index(
                index_name=index_sel,
                query=query,
                embed_model_path=embed_model,
                llm_model_name=llm_model,
                use_llm=llm_use,
                top_k=top_k,
            )

            if res.get("status") == "ok":
                st.success("検索完了")
                st.json(res.get("response"))
            else:
                st.error(f"エラー: {res.get('error')}")
