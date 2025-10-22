from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain.chains import retrieval_qa

from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager

import chromadb
import os
from typing import List, Optional
import json
import re
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)

# ------------ SET DOCKER ------------
os.environ["AUTOGEN_USE_DOCKER"] = "False"

LLM_CONFIG_70b = {
    "config_list": [
        {
            "model": "llama3:70b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "price": [0, 0],
        }
    ]
}

LLM_CONFIG_8b = {
    "config_list": [
        {
            "model": "llama3:8b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "price": [0, 0],
        }
    ]
}

COLLECTION_NAME = "collection_trial"
PERSISTED_DIR = "./chroma_db"

def load_chunk_persist_pdf() -> Chroma:
    EMBED_MODEL = "nomic-embed-text:v1.5"

    embedding = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=PERSISTED_DIR,
        embedding_function=embedding,
        collection_name=COLLECTION_NAME
    )
    print("chromadb realoaded successfully")
    return vectordb

def build_agents():
    vectordb = load_chunk_persist_pdf()
    # 1) User agent (entry point)
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
    )

    # 2) Retriever agent:
    retriever_agent = AssistantAgent(
        name="retriever",
        llm_config=LLM_CONFIG_8b,
        system_message=(
            "You are a retrieval orchestrator. "
            "When given a USER QUESTION, you DO NOT answer it directly. "
            "Instead, retrieve the most relevant chunks to the USER QUESTION from the knowledge base "
            "and produce a concise CONTEXT block containing only extracted quotes. "
            "Don't forget to mention the file path. "
            "Provided CONTEXT HAS TO ALIGN with the USER QUESTION. "
            "Answer concisely in Bahasa Indonesia if the user asks in Bahasa Indonesia. "
            "End your final message with TERMINATE."
        ),
    )

    # 3) validator chunks
    validator_agent = AssistantAgent(
        name="validator",
        llm_config=LLM_CONFIG_8b,
        system_message=(
            """
              You are a grader assessing relevance of a retrieved document to a user question.
              Here is the retrieved context: {context}
              Here is the user question: {question}
              If the user’s question mentions a file and the retrieved context comes from that file, return a binary score: yes
              If the user’s question explicitly mentions a file (e.g., perjanjian kerja v2) and the retrieved context comes from a different file (e.g., perjanjian_kerja_v1.pdf), return a binary score: no
              If the user’s question explicitly mentions a file (e.g., perjanjian kerja v2) and the retrieved context comes from a different file (e.g., perjanjian_kerja_v2.pdf), return a binary score: yes
              If you receive 'Tidak ada file yang ditemukan untuk pertanyaan ini.' don't give a binary answer, instead you answer 'Tidak ada file yang ditemukan untuk pertanyaan ini.'
              If the document contains keywords related to the user question, grade it as relevant.
              It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
              Give answer a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
              Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
                """
        )
    )

    # 4) QA agent (final answer)
    qa_agent = AssistantAgent(
        name="qa_agent",
        llm_config=LLM_CONFIG_70b,
        system_message=(
            """
                You are a precise QA assistant.
                Here is the user question: {question}
                Here is the context: {context}.
                If the context is 'Tidak ada file yang ditemukan untuk pertanyaan ini.', it means you cannot
                find related file. You shall answer based on the that.
                If the context is 'Tidak ditemukan jawaban karena tidak ditemukan konteks yang sesuai', it means you have no
                valid related context. You shall answer based on that. DO NOT USE YOUR GENERAL KNOWLEDGE.
                If the context is insufficient, explicitly say so and ask for clarification.
                Answer in Bahasa Indonesia if the question in Bahasa Indonesia.
                Always mention the SOURCE FILE that listed in the context.
                End your final message with TERMINATE.
            """
        ),
    )

    def state_transition(last_speaker, groupchat):
      messages = groupchat.messages

      if last_speaker is user:
          return groupchat.agents[1]
      elif last_speaker.name == "retriever":
          return groupchat.agents[2]
      elif last_speaker.name == "validator":
          return groupchat.agents[3]
      else:
        return None


    groupchat = GroupChat(
        agents=[user, retriever_agent, validator_agent, qa_agent],
        messages=[],
        max_round=50,
        speaker_selection_method=state_transition,
    )
    groupchat.shared_state = {} # to pass function return to the function between agents

    manager = GroupChatManager(groupchat=groupchat, llm_config=LLM_CONFIG_70b)


    def retrieval_hook(messages: List[dict]) -> Optional[str]:
        # Find latest user question in the conversation
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return None
        question = user_msgs[-1]["content"]
        # question_vector = embedding.embed_query(question)

        # get file inside the question
        list_filenames_to_look = re.findall(r'[\w\-.]+\.pdf', question, flags=re.IGNORECASE)
        ctx = []
        ctx_snippet = []

        if list_filenames_to_look: # if file path found in the question
            # print(list_filenames_to_look)
            client = chromadb.PersistentClient(PERSISTED_DIR)
            collection = client.get_collection(COLLECTION_NAME)
            ids = collection.get(
                where={"source": {"$in":list_filenames_to_look}},
                include=["metadatas"],
            )["ids"] # get all ids documents metadata
            if not ids:
                ctx_str = "CONTEXT:\n- Tidak ada file yang ditemukan untuk pertanyaan ini."
                ctx_snippet = ""
                return ctx_str, ctx_snippet, question
            else:
                for id in ids:
                    source = collection.get(ids=id, include=["metadatas"])["metadatas"][0]["source"]
                    page = collection.get(ids=id, include=["metadatas"])["metadatas"][0]["page"]
                    snippet = collection.get(ids=id, include=["documents"])["documents"][0]
                    snippet = f"This context is based on {source}: {snippet}"
                    ctx.append(f"- Source: {source}, Page: {page}\n  {snippet}")
                    ctx_snippet.append(snippet) # Append as string
                    ctx_str = "\n".join(ctx)
                return ctx_str, ctx_snippet, question
        else:
            # retriever = vectordb.similarity_search(question, k=2)
            retriever = vectordb.max_marginal_relevance_search(question, k=15)
            # print("woy ini retriever: ",retriever)
            if not retriever:
                # print(f"[Retriever] No results for query: {question}")
                ctx_str = "CONTEXT:\n- Tidak ada konteks yang ditemukan untuk pertanyaan ini."
                ctx_snippet = ""
                return ctx_str, ctx_snippet, question
            else:
                for r in retriever:
                    source = r.metadata.get("source", "unknown file")
                    page = r.metadata.get("page", "unknown page")
                    snippet = r.page_content.strip().replace("\n", " ")
                    snippet = f"This context is based on {source}: {snippet}"
                    ctx.append(f"- Source: {source}, Page: {page}\n  {snippet}")
                    ctx_snippet.append(snippet) # Append as string
                    ctx_str = "\n".join(ctx)
                return ctx_str, ctx_snippet, question

    orig_retriever_generate_reply = retriever_agent.generate_reply
    def retriever_generate_reply(*args, **kwargs):
        # Build CONTEXT and send it as the retriever's content
        ctx_str, ctx_snippet, question = retrieval_hook(groupchat.messages)
        groupchat.shared_state["retrieve_ctx"] = {
            "ctx_str":ctx_str,
            "ctx_snippet":ctx_snippet,
            "question":question,
        }

        if ctx_str:
            # Return the context string as the message content
            return {"role": "assistant", "content": ctx_str, "name": retriever_agent.name}
        # If anything goes odd, fall back to normal behavior
        return orig_retriever_generate_reply(*args, **kwargs)
    retriever_agent.generate_reply = retriever_generate_reply


    orig_validator_generate_reply = validator_agent.generate_reply
    def validator_hook(*args, **kwargs):
        logging.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

        retrievel_data = groupchat.shared_state.get("retrieve_ctx", {})
        question = retrievel_data.get("question")
        ctx_snippet = retrievel_data.get("ctx_snippet", [])  # list of strings
        ctx_str = retrievel_data.get("ctx_str", [])  # list of strings
        # print("jumlah ctx_snippet: ", len(ctx_snippet))

        filtered_docs = []
        if ctx_snippet:
            for d in ctx_snippet:
                logging.info(f"validate this snippet: {d}")
                validator_agent_sys_msg = validator_agent.system_message.format(
                    context=d,
                    question=question,
                )
                msg = [{"role": "system", "content": validator_agent_sys_msg}]
                raw = orig_validator_generate_reply(msg)

                # normalize raw -> content string
                if isinstance(raw, dict) and "content" in raw:
                    content = raw["content"]
                else:
                    content = str(raw)

                # try to extract JSON object if LLM wrapped it with extra text
                m = re.search(r'\{.*\}', content, flags=re.DOTALL)
                json_text = m.group(0) if m else content

                try:
                    score = json.loads(json_text)
                    grade = score.get("score", "").lower()
                    logging.info(grade)
                except Exception:
                    logging.info("---GRADE: COULD NOT PARSE OUTPUT---")
                    # print("Raw output:", content)
                    grade = "no"
                    logging.info(grade)

                if grade == "yes":
                    # filtered_docs.append(f"- {d}\n")
                    filtered_docs.append(f"{d}")

            # save validator result for downstream QA agent
            # return a proper assistant message dict (autogen expects this)
            content_to_return = "\n".join(filtered_docs) if filtered_docs else "Tidak ditemukan jawaban karena tidak ditemukan konteks yang sesuai"
            groupchat.shared_state["validator_ctx"] = {
                "filtered_ctx": content_to_return,
                "question": question,
            }
            return {"role": "assistant", "content": content_to_return, "name": validator_agent.name}
        else:
            # no context found -> ask validator to follow its 'no file' rule
            validator_agent_sys_msg = validator_agent.system_message.format(
                context=ctx_str,
                question=question,
            )
            msg = [{"role": "system", "content": validator_agent_sys_msg}]
            raw = orig_validator_generate_reply(msg)

            if isinstance(raw, dict) and "content" in raw:
                content = raw["content"]
            else:
                content = str(raw)

            try:
                score = json.loads(content)
                answer = score.get("score", "").lower()
            except Exception:
                answer = content

            groupchat.shared_state["validator_ctx"] = {
                "filtered_ctx": answer,
                "question": question,
            }

            return {"role": "assistant", "content": answer, "name": validator_agent.name}
    validator_agent.generate_reply = validator_hook


    orig_qa_generate_reply = qa_agent.generate_reply
    def qa_hook(*args, **kwargs):
        validator_ctx = groupchat.shared_state.get("validator_ctx", {})
        question = validator_ctx.get("question")
        filtered_ctx = validator_ctx.get("filtered_ctx")  # list or string
        print("ini filtered_ctx: ",filtered_ctx)

        # print("ini question ", question)
        # print("ini filtered_ctx".join(filtered_ctx))

        qa_agent_sys_msg = qa_agent.system_message.format(
            context="\n".join(filtered_ctx) if isinstance(filtered_ctx, list) else filtered_ctx,
            # context="\n".join(filtered_ctx) if filtered_ctx else "Tidak ditemukan jawaban karena tidak ditemukan konteks yang sesuai",
            question=question,
        )
        msg = [{"role":"system", "content": qa_agent_sys_msg}]
        print(f"ini msg: {msg}")
        raw = orig_qa_generate_reply(msg)

        if isinstance(raw, dict) and "content" in raw:
            content = raw["content"]
        else:
            content = str(raw)

        # optionally persist final answer
        groupchat.shared_state["final_answer"] = {
            "question": question,
            "content": content,
        }

        return {"role": "assistant", "content": content, "name": qa_agent.name}

    qa_agent.generate_reply = qa_hook


    return user, manager

# Streamlit App
st.set_page_config(page_title="Agentic RAG Trial")
st.title("Document QA Assistant Trial")

# Cache the agents to prevent re-initialization on every rerun
user, manager = build_agents()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask me !")
# if prompt := st.chat_input("Ask me !"):
if prompt and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # The core logic: initiate chat and get the final answer
        # Create a container for the QA agent's final answer
        answer_placeholder = st.empty()
        answer_placeholder.markdown("⏳ Processing...")

        # Start the chat with the user's prompt
        user.initiate_chat(manager, message=prompt)

        # Retrieve the final answer from the shared state
        final_answer = manager.groupchat.shared_state.get("final_answer", {}).get("content")

        if final_answer:
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            answer_placeholder.markdown(final_answer)
        else:
            st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I couldn't find an answer."})
            answer_placeholder.markdown("I'm sorry, I couldn't find an answer.")