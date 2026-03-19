import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# Set up the page title
st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")
st.title("🎥 YouTube Transcript Chatbot")

# --- SIDEBAR FOR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    hf_token = st.text_input("HuggingFace API Token", type="password", help="Enter your HF token to use Llama-3.1")
    st.markdown("This chatbot uses `meta-llama/Llama-3.1-8B-Instruct` via the Hugging Face Endpoint.")

# --- SESSION STATE INITIALIZATION ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- MAIN APP: VIDEO PROCESSING ---
st.subheader("1. Load Video Transcript")
video_id = st.text_input("Enter YouTube Video ID (e.g., Gfr50f6ZBvo)")

if st.button("Process Video"):
    if not video_id:
        st.warning("Please enter a Video ID.")
    else:
        with st.spinner("Fetching transcript and building vector store (this may take a minute)..."):
            try:
                # 1. Fetch transcript (From your notebook)
                ytt_api = YouTubeTranscriptApi()
                fetched_transcript = ytt_api.fetch(video_id)
                transcript_text = " ".join(chunk['text'] for chunk in fetched_transcript)
                
                # 2. Text Splitter (From your notebook)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript_text])
                
                # 3. Embedding Generation & Vector Store Indexing
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(chunks, embeddings)
                
                # Save FAISS vector store to session state so it persists
                st.session_state.vector_store = vector_store
                st.success(f"Video `{video_id}` processed successfully! {len(chunks)} chunks created.")
            except Exception as e:
                st.error(f"Error processing video transcript: {e}")

# --- MAIN APP: CHAT INTERFACE ---
st.subheader("2. Ask Questions")
question = st.text_input("Ask a question based on the video's context:")

if st.button("Get Answer"):
    if not hf_token:
        st.error("Please enter your Hugging Face API Token in the sidebar.")
    elif not st.session_state.vector_store:
        st.error("Please process a video first before asking questions.")
    elif not question:
        st.warning("Please type a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                # 1. Setup LLM 
                repo_id = "meta-llama/Llama-3.1-8B-Instruct"
                llm = HuggingFaceEndpoint(
                    repo_id=repo_id,
                    task="text-generation",
                    max_new_tokens=512,
                    temperature=0.7,
                    huggingfacehub_api_token=hf_token,
                )
                model = ChatHuggingFace(llm=llm)
                
                # 2. Retrieve Context via FAISS
                retriever = st.session_state.vector_store.as_retriever()
                retrieved_docs = retriever.invoke(question)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                
                # 3. Construct Prompt (From your notebook)
                prompt_template = """You are a helpful assistant.
Answer only from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}"""
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=['context', 'question']
                )
                
                final_prompt = prompt.format(context=context_text, question=question)
                
                # 4. Generate Answer
                answer = model.invoke(final_prompt)
                
                # 5. Display Answer
                st.write("### Answer:")
                st.info(answer.content)
                
                # Optional: Show expandable sources
                with st.expander("View Retrieved Context Chunks"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1}:**\n {doc.page_content}")
                        
            except Exception as e:
                st.error(f"Error generating answer. Check your HF token. Details: {e}")