import os
import warnings
import streamlit as st
from dotenv import load_dotenv

# Suppress all warnings and disable caching BEFORE any langchain imports
warnings.filterwarnings('ignore')
os.environ['LANGCHAIN_VERBOSE'] = 'false'
os.environ['LANGCHAIN_DEBUG'] = 'false'
os.environ['LANGCHAIN_CACHE'] = 'false'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Initialize langchain module FIRST before any LangChain imports
import langchain
langchain.verbose = False
langchain.debug = False
try:
    langchain.llm_cache = None
except:
    pass

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

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
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">🧠 MATLAB Financial Toolbox RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions about MATLAB Financial Toolbox with AI-powered retrieval</div>', unsafe_allow_html=True)

# Sidebar for configuration and info
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check API keys
    pinecone_key = os.getenv("PINECONE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if pinecone_key and openai_key and index_name:
        st.success("✓ All API keys configured")
        st.info(f"📍 Pinecone Index: `{index_name}`")
    else:
        st.error("❌ Missing API keys")
        st.warning("Please configure PINECONE_API_KEY, OPENAI_API_KEY, and PINECONE_INDEX_NAME in .env file")
    
    st.divider()
    
    st.subheader("🔧 Model Settings")
    top_k = st.slider("Number of context chunks to retrieve:", 1, 10, 5, help="How many relevant sections to consider")
    num_context = st.slider("Context chunks to use in answer:", 1, top_k, 3, help="How many of the retrieved chunks to include in the prompt")
    temperature = st.slider("Temperature (creativity):", 0.0, 1.0, 0.0, help="Lower = more deterministic, Higher = more creative")
    
    st.divider()
    
    st.subheader("📚 About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) assistant:
    - Searches MATLAB Financial Toolbox documentation
    - Retrieves relevant sections using embeddings
    - Generates answers using GPT-4o
    
    **Built with:**
    - 🔍 Pinecone (vector database)
    - 🤖 OpenAI (embeddings & LLM)
    - 🔗 LangChain (orchestration)
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
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - How to do matrix multiplication in MATLAB?
        - How to do matrix division in MATLAB?
        - How to estimate transition probabilities using Financial Toolbox?
        - How to install MATLAB?
        - How to perform matrix operations in MATLAB?
        """)

with col2:
    st.markdown("### ⚡ Quick Actions")
    reset_btn = st.button("🔄 Reset", use_container_width=True)
    if reset_btn:
        st.rerun()

# Submit button
submit_btn = st.button("🚀 Get Answer", type="primary", use_container_width=True)

# Process query
if submit_btn and query.strip():
    try:
        with st.spinner("🔄 Processing your question..."):
            
            # Step 1: Initialize Pinecone
            st.status("🔗 Connecting to Pinecone...", state="running")
            try:
                pc = Pinecone(api_key=pinecone_key)
                index = pc.Index(index_name)
                st.status("🔗 Connected to Pinecone", state="complete")
            except Exception as e:
                st.error(f"Failed to connect to Pinecone: {e}")
                st.stop()
            
            # Step 2: Initialize embeddings and LLM
            st.status("🤖 Initializing models...", state="running")
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                llm = ChatOpenAI(model="gpt-4o", temperature=temperature)
                st.status("🤖 Models initialized", state="complete")
            except Exception as e:
                st.error(f"Failed to initialize models: {e}")
                st.stop()
            
            # Step 3: Generate query embedding
            st.status("📊 Generating query embedding...", state="running")
            try:
                query_embedding = embeddings.embed_query(query)
                st.status("📊 Query embedding generated", state="complete")
            except Exception as e:
                st.error(f"Failed to generate embedding: {e}")
                st.stop()
            
            # Step 4: Search Pinecone
            st.status(f"🔍 Searching Pinecone (top {top_k})...", state="running")
            try:
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace="production"
                )
                st.status(f"🔍 Found {len(results.get('matches', []))} results", state="complete")
            except Exception as e:
                st.error(f"Pinecone search failed: {e}")
                st.stop()
            
            # Step 5: Extract context
            st.status("📋 Extracting context...", state="running")
            context_parts = []
            for match in results.get('matches', []):
                if 'metadata' in match and 'text' in match['metadata']:
                    text = match['metadata']['text']
                    context_parts.append(text)
            
            context = "\n\n".join(context_parts[:num_context])
            st.status(f"📋 Extracted {len(context_parts[:num_context])} context sections", state="complete")
            
            if not context:
                st.warning("⚠️ No relevant documents found in the knowledge base")
                st.stop()
            
            # Step 6: Generate answer
            st.status("💭 Generating answer with GPT-4o...", state="running")
            try:
                prompt_template = """Use the following pieces of context to answer the question at the end.
Answer only based on the context given.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
                
                prompt = PromptTemplate.from_template(prompt_template)
                formatted_prompt = prompt.format(context=context, question=query)
                
                response = llm.invoke([HumanMessage(content=formatted_prompt)])
                answer = response.content
                st.status("💭 Answer generated", state="complete")
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
                st.stop()
            
            # Display results
            st.success("✅ Query processed successfully!")
            
            # Query section
            st.markdown("---")
            st.markdown('<div class="query-box">', unsafe_allow_html=True)
            st.markdown(f"**🎯 Your Question:**\n\n{query}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Answer section
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(f"**✨ Answer:**\n\n{answer}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Context section
            with st.expander("📚 Context Sources (Click to view)", expanded=False):
                st.markdown('<div class="context-box">', unsafe_allow_html=True)
                for i, ctx in enumerate(context_parts[:num_context], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                    st.divider()
                st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        import traceback
        with st.expander("📋 Error Details"):
            st.code(traceback.format_exc())

elif submit_btn and not query.strip():
    st.warning("Please enter a question first!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>MATLAB Financial Toolbox RAG Assistant | Powered by OpenAI, Pinecone & LangChain</p>
</div>
""", unsafe_allow_html=True)
