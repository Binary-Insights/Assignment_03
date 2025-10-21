"""
Streamlit Frontend for Enhanced MATLAB RAG Assistant
Pure UI layer - communicates with FastAPI backend
Supports both original RAG and enhanced RAG (with Wikipedia fallback & structured output)
Location: src/frontend/rag_streamlit_frontend.py
"""

import streamlit as st
import requests
import json
from typing import Optional
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced MATLAB RAG Assistant",
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
st.markdown('<div class="main-header">🧠 Enhanced MATLAB RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions about MATLAB with AI-powered retrieval, Wikipedia fallback, and structured knowledge extraction</div>', unsafe_allow_html=True)

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
    use_vector_db = st.checkbox("Use Vector DB Search", value=True, help="Search Pinecone vector database first")
    use_wikipedia = st.checkbox("Allow Wikipedia Fallback", value=True, help="Search Wikipedia if not found in vector DB")
    
    st.subheader("📊 Vector DB Results")
    num_vector_results = st.slider(
        "Number of Vector DB results to use for structured note:",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Using more results provides richer context but may be slower. Set to 1 to use only top result."
    )
    
    st.caption(f"📌 Currently using top-{num_vector_results} result(s) from Pinecone")
    
    st.divider()
    
    st.subheader("📚 About This App")
    st.markdown("""
    This Enhanced RAG assistant:
    - 🔍 **Vector Search**: Searches Pinecone vector database
    - 📚 **Wikipedia Fallback**: Falls back to Wikipedia if not found
    - 🗄️ **Smart Caching**: Caches concepts in PostgreSQL for faster retrieval
    - 📊 **Structured Output**: Uses Instructor to generate consistent structured notes
    
    **3-Tier Search Strategy:**
    1. **Pinecone Vector DB** → Direct knowledge base search
    2. **PostgreSQL Cache** → Pre-computed cached results (20-40x faster!)
    3. **Wikipedia** → Fallback source for new concepts
    
    **Architecture:**
    - 🎨 **Frontend**: Streamlit (this app)
    - ⚙️ **Backend**: FastAPI server (enhanced_fastapi_server.py)
    - 🔗 **Communication**: HTTP REST API
    - 📦 **Databases**: Pinecone + PostgreSQL
    
    **Built with:**
    - OpenAI (embeddings & LLM)
    - Pinecone (vector database)
    - PostgreSQL (concept caching)
    - Instructor (structured outputs)
    - Wikipedia API (fallback source)
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
                "use_vector_db": use_vector_db,
                "use_wikipedia": use_wikipedia,
                "num_vector_results": num_vector_results
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
            
            # Metadata section
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                source_label = result.get('source', 'unknown').upper()
                st.metric("📍 Source", source_label)
            with col2:
                is_cached = "✓ Cached" if result.get('cached') else "✗ Not Cached"
                st.metric("💾 Cache", is_cached)
            with col3:
                processing_time = result.get('processing_time_ms', 0)
                st.metric("⏱️ Time (ms)", f"{processing_time:.0f}")
            with col4:
                found_vector = "✓ Found" if result.get('concept_found_in_vector') else "✗ Not Found"
                st.metric("🔍 Vector DB", found_vector)
            
            # Structured Note section
            st.markdown("---")
            st.markdown("### 📚 Structured Knowledge Note")
            
            structured_note = result.get('structured_note', {})
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Definition", "Characteristics", "Applications", "Related Concepts", "Raw JSON"])
            
            with tab1:
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                if 'definition' in structured_note:
                    definition = structured_note['definition']
                    st.markdown(f"**Primary Definition:**\n\n{definition.get('primary', 'N/A')}")
                    if definition.get('alternative'):
                        st.markdown(f"\n**Alternative Definition:**\n\n{definition.get('alternative', 'N/A')}")
                    if definition.get('context'):
                        st.markdown(f"\n**Context:**\n\n{definition.get('context', 'N/A')}")
                    
                    # Display code examples if available
                    if definition.get('code_examples'):
                        st.markdown("\n---")
                        st.markdown("**MATLAB Code Examples:**")
                        for i, code_example in enumerate(definition['code_examples'], 1):
                            st.code(code_example, language="matlab")
                    
                    # Display example explanation if available
                    if definition.get('example_explanation'):
                        st.markdown("\n**Code Explanation:**")
                        st.markdown(definition['example_explanation'])
                else:
                    st.warning("No definition available")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="context-box">', unsafe_allow_html=True)
                if 'key_characteristics' in structured_note:
                    chars = structured_note['key_characteristics']
                    if chars.get('characteristics'):
                        st.markdown("**Key Characteristics:**")
                        for i, char in enumerate(chars['characteristics'], 1):
                            st.markdown(f"  {i}. {char}")
                    if chars.get('importance'):
                        st.markdown(f"\n**Importance:** {chars['importance']}")
                else:
                    st.warning("No characteristics available")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="context-box">', unsafe_allow_html=True)
                if 'applications' in structured_note:
                    apps = structured_note['applications']
                    if apps.get('use_cases'):
                        st.markdown("**Use Cases:**")
                        for i, use_case in enumerate(apps['use_cases'], 1):
                            st.markdown(f"  {i}. {use_case}")
                    if apps.get('industry_examples'):
                        st.markdown("\n**Industry Examples:**")
                        for i, example in enumerate(apps['industry_examples'], 1):
                            st.markdown(f"  {i}. {example}")
                    if apps.get('matlab_relevance'):
                        st.markdown(f"\n**MATLAB Relevance:** {apps['matlab_relevance']}")
                else:
                    st.warning("No applications available")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown('<div class="context-box">', unsafe_allow_html=True)
                if 'related_concepts' in structured_note:
                    related = structured_note['related_concepts']
                    if related.get('related_terms'):
                        st.markdown("**Related Terms:**")
                        for i, term in enumerate(related['related_terms'], 1):
                            st.markdown(f"  {i}. {term}")
                    if related.get('relationships'):
                        st.markdown("\n**Relationships:**")
                        for i, rel in enumerate(related['relationships'], 1):
                            st.markdown(f"  {i}. {rel}")
                else:
                    st.warning("No related concepts available")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab5:
                st.json(structured_note)
            
            # Source context section
            st.markdown("---")
            with st.expander("📖 Source Context", expanded=False):
                if result.get('source') == 'vector_db' and result.get('pinecone_context'):
                    st.markdown("**Context from Vector DB:**")
                    st.markdown(f"**Number of Results Used**: {num_vector_results}")
                    st.markdown("---")
                    # Display full context without truncation
                    st.text_area(
                        "Vector DB Results:",
                        value=result['pinecone_context'],
                        height=300,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                elif result.get('source') == 'wikipedia' and result.get('wikipedia_context'):
                    st.markdown("**Context from Wikipedia:**")
                    st.text_area(
                        "Wikipedia Context:",
                        value=result['wikipedia_context'],
                        height=300,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                else:
                    st.info("No source context available")
            
            # Confidence score
            if structured_note.get('confidence_score'):
                confidence = structured_note['confidence_score']
                st.markdown(f"**Confidence Score:** {confidence:.1%}")
                st.progress(confidence)

    
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to FastAPI server at {st.session_state.api_url}")
        st.info("Make sure the backend server is running:")
        st.code("python enhanced_fastapi_server.py")
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
    <p>Enhanced MATLAB RAG Assistant with Wikipedia Fallback & PostgreSQL Caching</p>
    <p>Frontend: Streamlit | Backend: FastAPI | Vector DB: Pinecone | Cache: PostgreSQL | LLM: OpenAI</p>
</div>
""", unsafe_allow_html=True)
