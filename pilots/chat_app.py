import streamlit as st
import requests

# Set page configuration
st.set_page_config(page_title="ChatBot", page_icon="🤖", layout="centered")

# Title
st.title("🤖 ChatBot with OpenAI")

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/chat"

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json={"message": prompt})
                response_data = response.json()
                
                if "reply" in response_data:
                    reply = response_data["reply"]
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                elif "error" in response_data:
                    error_msg = f"Error: {response_data['error']}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = "Unexpected response format"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"Failed to connect to API: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.info("This chatbot uses OpenAI's API via FastAPI backend.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
