import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="YouTube AI Chat", page_icon="🤖", layout="wide")

st.markdown(
    """
<h1 style='text-align:center;'>🤖 Chat with YouTube Video</h1>
<p style='text-align:center;'>Ask questions about any YouTube video using AI</p>
""",
    unsafe_allow_html=True,
)

# ------------------ SESSION STATE ------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ SIDEBAR ------------------
with st.sidebar:

    st.title("⚙️ Settings")

    hf_token = st.text_input("HuggingFace Token", type="password")

    model_choice = st.selectbox(
        "Select Model",
        ["HuggingFaceH4/zephyr-7b-beta", "meta-llama/Llama-3.1-8B-Instruct"],
    )

    st.divider()

    st.subheader("📺 YouTube Video")

    video_id = st.text_input("Enter Video ID", placeholder="example: Gfr50f6ZBvo")

    if video_id:
        st.video(f"https://www.youtube.com/watch?v={video_id}")

    process = st.button("Process Video")

    if process:

        with st.spinner("Processing video..."):

            try:

                ytt_api = YouTubeTranscriptApi()
                fetched_transcript = ytt_api.fetch(video_id)

                transcript = " ".join(chunk.text for chunk in fetched_transcript)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )

                docs = splitter.create_documents([transcript])

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_store = FAISS.from_documents(docs, embeddings)

                st.success("Video processed successfully!")

                st.session_state.messages = []

            except Exception as e:
                st.error(e)

    if st.button("Clear Chat"):
        st.session_state.messages = []

# ------------------ CHAT UI ------------------

chat_container = st.container()

with chat_container:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ------------------ CHAT INPUT ------------------

user_prompt = st.chat_input("Ask anything about this video...")

if user_prompt:

    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):

        if not hf_token:
            st.error("Please add HuggingFace token in sidebar")

        elif not st.session_state.vector_store:
            st.error("Process a video first")

        else:

            with st.spinner("AI is thinking..."):

                try:

                    llm = HuggingFaceEndpoint(
                        repo_id=model_choice,
                        task="text-generation",
                        max_new_tokens=512,
                        temperature=0.7,
                        huggingfacehub_api_token=hf_token,
                    )

                    model = ChatHuggingFace(llm=llm)

                    retriever = st.session_state.vector_store.as_retriever()

                    docs = retriever.invoke(user_prompt)

                    context = "\n\n".join(doc.page_content for doc in docs)

                    prompt_template = """
You are an AI assistant.

Answer only from the transcript context.

Context:
{context}

Question:
{question}
"""

                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"],
                    )

                    final_prompt = prompt.format(context=context, question=user_prompt)

                    response = model.invoke(final_prompt)

                    st.markdown(response.content)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.content}
                    )

                except Exception as e:
                    st.error(e)
