from openai import OpenAI
import streamlit as st
from rag import ChromaRag
from langchain_openai import OpenAIEmbeddings

st.title("AI Doctor")

rag = ChromaRag(
    persist_directory=r"chroma",
    embedding_fun=OpenAIEmbeddings(model="text-embedding-3-large"),
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    results = rag.similarity_search(query=prompt, k=10)
    st.session_state.messages[-1]["rag"] = results
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": f'RAG Results: {m["rag"]}, User Query: {m["content"]}'}
                for m in st.session_state.messages[-5:]
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})