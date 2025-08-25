# Best ChatGPT-style UI with Streamlit
import streamlit as st
from query import search_and_answer

st.set_page_config(page_title="Workover Chat", page_icon="💬")
st.title("💬 Workover Report Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get user input
user_input = st.chat_input("Ask your question...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = search_and_answer(user_input)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
