from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import os
import chromadb

# CHUNK_SIZE = 200
# CHUNK_OVERLAP = 10
EMBED_MODEL = "nomic-embed-text:v1.5"
COLLECTION_NAME = "hr_collection_trial"
PERSISTED_DIR = "chroma_db"
FOLDER_PATH = "document_hr/"

try:
    embedding = OllamaEmbeddings(model=EMBED_MODEL)
    text_splitter = SemanticChunker(embeddings=embedding)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP
    # )
    all_chunked_documents = []
    cnt = 0
    for file in os.listdir(FOLDER_PATH):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(FOLDER_PATH, file)
            loader = PyMuPDFLoader(pdf_path)
            document = loader.load()
            for d in document:
                d.metadata["source"] = file
                # all_chunked_documents.append(d)
            chunked_document = text_splitter.split_documents(document)
            all_chunked_documents.extend(chunked_document)
            cnt += 1
    print(f"{cnt} PDFs completely ingested !")
    # for doc in all_chunked_documents:
    #     doc.page_content = f"search_document: {doc.page_content}"
    # client = chromadb.PersistentClient(PERSISTED_DIR)
    # if client.list_collections():
    # client.get_or_create_collection(COLLECTION_NAME)
    # else:
    #     print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=all_chunked_documents,
        embedding=embedding,
        persist_directory=PERSISTED_DIR,
        collection_name=COLLECTION_NAME
    )
    print(f"total chunks: {vectordb._collection.count()}")
    print("chromadb SUCCESSFULLY ingested.")
except Exception as e:
    print(f"error: {e}")
    print("chromadb UNSUCCESSFULLY ingested.")