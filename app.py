import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO = "shahab1/qwen-tutor-gguf"
MODEL_FILE = "tutor-q4_k_m.gguf"
SYSTEM_PROMPT = "You are a friendly offline tutor for school students. Explain simply and step by step."

st.set_page_config(page_title="Offline Tutor", page_icon="📚")
st.title("📚 Offline Tutor")
st.caption("Ask a school Math or Physics question.")


@st.cache_resource(show_spinner="Loading model (first time takes a few minutes)...")
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return Llama(
        model_path=path,
        n_ctx=1024,      # small context = less RAM
        n_batch=128,
        n_threads=2,
        chat_format="chatml",
        verbose=False,
    )


llm = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

question = st.chat_input("Type your question...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question[:1000]},
    ]

    def stream():
        for chunk in llm.create_chat_completion(
            messages=messages, max_tokens=350, temperature=0.2, stream=True
        ):
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta

    with st.chat_message("assistant"):
        answer = st.write_stream(stream())
    st.session_state.history.append(("assistant", answer))
