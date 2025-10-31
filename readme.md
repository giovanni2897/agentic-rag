🧠 Agentic RAG System

![image_alt](https://github.com/giovanni2897/agentic-rag/blob/22ab6c4be3364f1d5d6cb2b213c8e85b377917f8/pics/UI_streamlit.png?raw=true)

This repository implements an Agentic Retrieval-Augmented Generation (RAG) system designed to run fully locally using Ollama. The system combines multi-agent collaboration, local embeddings, and vector-based retrieval to produce accurate and validated answers.

⚙️ Overview

The project is structured into three main files:
* ingest2.py — Handles data ingestion into the vector database.
* app6_st.py — Runs the Agentic RAG application using Streamlit for the UI.
* requirements_noversion.txt - requirement libraries lies here.

* Vector DB: ChromaDB

* Embedding model: nomic-embed-text:v1.5 (from Ollama)


Framework: AutoGen

Models:
* Qwen3:32b for the QA agent
* Llama3:8b for the validator agent

All models are retrieved from Ollama, ensuring everything runs 100% locally.

🧩 Agentic Architecture

This RAG system is built around three collaborative agents:

* Retriever Agent
Retrieves relevant context from the vector database (ChromaDB) based on the user’s question.

* Validator Agent
Uses Llama3:8b to validate the retrieved context and ensure alignment between the context and the question.

* QA Agent
Uses Qwen3:8b to generate the final answer based on the validated context provided by the Validator Agent.

🚀 Key Features
* Runs entirely on local environment (no external API calls)
* Privacy-first architecture — no data leaves your machine
* Multi-agent reasoning powered by AutoGen
* Context-aware retrieval using ChromaDB and Nomic embeddings
* Interactive UI built with Streamlit