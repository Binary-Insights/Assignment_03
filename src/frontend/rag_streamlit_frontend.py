"""
Streamlit Frontend for MATLAB RAG Assistant
Pure UI layer - communicates with FastAPI backend
Location: pilots/rag_streamlit_frontend.py
"""

import streamlit as st
import requests
import json
from typing import Optional

# Configure Streamlit page
st.set_page_config(
    page_title="MATLAB RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2em;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin-top: 20px;
    }
    .context-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        margin-top: 15px;
        font-size: 0.9em;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-healthy {
        background-color: #e8f5e9;
        color: #2e7d32;
        border-left: 4px solid #2e7d32;
    }
    .status-unhealthy {
        background-color: #ffebee;
        color: #c62828;
        border-left: 4px solid #c62828;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for API URL and responses
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if 'last_response' not in st.session_state:
    st.session_state.last_response = None

# Title and description
st.markdown('<div class="main-header">🧠 MATLAB Financial Toolbox RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions about MATLAB Financial Toolbox with AI-powered retrieval</div>', unsafe_allow_html=True)

# Sidebar for configuration and info
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API URL configuration
    st.subheader("🔌 API Configuration")
    api_url = st.text_input(
        "FastAPI Server URL:",
        value=st.session_state.api_url,
        help="URL where FastAPI backend is running"
    )
    st.session_state.api_url = api_url
    
    # Check API health
    if st.button("🔍 Check API Health", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.markdown(
                    '<div class="status-box status-healthy">✓ API is healthy and running</div>',
                    unsafe_allow_html=True
                )
                config_response = requests.get(f"{api_url}/config", timeout=5)
                if config_response.status_code == 200:
                    config = config_response.json()
                    st.json(config)
            else:
                st.error(f"API returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.markdown(
                '<div class="status-box status-unhealthy">✗ Cannot connect to API. Is the server running?</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error checking API: {e}")
    
    st.divider()
    
    # Model Settings
    st.subheader("🔧 Model Settings")
    top_k = st.slider(
        "Number of context chunks to retrieve:",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant sections to consider from the knowledge base"
    )
    
    num_context = st.slider(
        "Context chunks to use in answer:",
        min_value=1,
        max_value=top_k,
        value=min(3, top_k),
        help="How many of the retrieved chunks to include in the LLM prompt"
    )
    
    temperature = st.slider(
        "Temperature (creativity):",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Lower = more deterministic, Higher = more creative"
    )
    
    st.divider()
    
    st.subheader("📚 About This App")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) assistant:
    - 🔍 Searches MATLAB Financial Toolbox documentation
    - 📊 Retrieves relevant sections using embeddings
    - 🤖 Generates answers using GPT-4o
    
    **Architecture:**
    - 🎨 **Frontend**: Streamlit (this app)
    - ⚙️ **Backend**: FastAPI server
    - 🔗 **Communication**: HTTP REST API
    
    **Built with:**
    - OpenAI (embeddings & LLM)
    - Pinecone (vector database)
    - LangChain (orchestration)
    """)


# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Ask a Question")
    
    # Query input
    query = st.text_area(
        "Enter your question about MATLAB:",
        placeholder="e.g., How to perform matrix multiplication in MATLAB?",
        height=100,
        label_visibility="collapsed"
    )
    
    # Example queries expander
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - How to do matrix multiplication in MATLAB?
        - How to do matrix division in MATLAB?
        - How to estimate transition probabilities using Financial Toolbox?
        - How to install MATLAB?
        - How to perform matrix operations in MATLAB?
        - What is the Financial Toolbox in MATLAB?
        - How to use the Optimization Toolbox?
        """)

with col2:
    st.markdown("### ⚡ Quick Actions")
    col_reset, col_info = st.columns(2)
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.last_response = None
            st.rerun()
    with col_info:
        if st.button("ℹ️ API Docs", use_container_width=True):
            st.info(f"Open {st.session_state.api_url}/docs in your browser")

# Submit button
submit_btn = st.button("🚀 Get Answer", type="primary", use_container_width=True)

# Process query
if submit_btn and query.strip():
    try:
        with st.spinner("🔄 Processing your question..."):
            
            # Prepare request
            st.status("📤 Sending request to FastAPI server...", state="running")
            request_payload = {
                "query": query,
                "top_k": top_k,
                "num_context": num_context,
                "temperature": temperature
            }
            
            # Make API call
            response = requests.post(
                f"{st.session_state.api_url}/query",
                json=request_payload,
                timeout=30
            )
            
            if response.status_code != 200:
                st.error(f"❌ API Error: {response.status_code}")
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"Details: {error_detail}")
                st.stop()
            
            st.status("📤 Request successful", state="complete")
            
            # Parse response
            result = response.json()
            st.session_state.last_response = result
            
            # Display results
            st.success("✅ Query processed successfully!")
            
            # Query section
            st.markdown("---")
            st.markdown('<div class="query-box">', unsafe_allow_html=True)
            st.markdown(f"**🎯 Your Question:**\n\n{result['query']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Answer section
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(f"**✨ Answer:**\n\n{result['answer']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Metadata
            st.info(f"📊 Sources Retrieved: {result['num_sources_retrieved']} | Model: {result['model_used']}")
            
            # Context section
            with st.expander("📚 Context Sources (Click to view)", expanded=False):
                st.markdown('<div class="context-box">', unsafe_allow_html=True)
                sources = result.get('sources', [])
                if sources:
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:**")
                        source_text = source.get('text', 'No text available')
                        display_text = source_text[:500] + "..." if len(source_text) > 500 else source_text
                        st.text(display_text)
                        if source.get('score'):
                            st.caption(f"Score: {source['score']:.4f}")
                        st.divider()
                else:
                    st.warning("No sources available")
                st.markdown('</div>', unsafe_allow_html=True)
    
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to FastAPI server at {st.session_state.api_url}")
        st.info("Make sure the backend server is running:")
        st.code("python -m uvicorn backends.rag_fastapi_server:app --reload --host localhost --port 8000")
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The server may be overloaded.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        with st.expander("📋 Error Details"):
            st.code(str(e))

elif submit_btn and not query.strip():
    st.warning("⚠️ Please enter a question first!")

# Display last response if available
if st.session_state.last_response and not submit_btn:
    st.divider()
    st.markdown("### 📋 Last Query")
    last = st.session_state.last_response
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Query: {last['query'][:50]}...")
    with col2:
        if st.button("🔄 Clear History"):
            st.session_state.last_response = None
            st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>MATLAB Financial Toolbox RAG Assistant</p>
    <p>Frontend: Streamlit | Backend: FastAPI | Database: Pinecone | LLM: OpenAI</p>
</div>
""", unsafe_allow_html=True)
