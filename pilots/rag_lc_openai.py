import os
import warnings
import sys

# Suppress all warnings and disable caching BEFORE any langchain imports
warnings.filterwarnings('ignore')
os.environ['LANGCHAIN_VERBOSE'] = 'false'
os.environ['LANGCHAIN_DEBUG'] = 'false'
os.environ['LANGCHAIN_CACHE'] = 'false'
os.environ['PYTHONWARNINGS'] = 'ignore'

from dotenv import load_dotenv

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

load_dotenv()


def main():
    print("="*70)
    print("MATLAB Financial Toolbox RAG - Pinecone + OpenAI")
    print("="*70)
    
    # Initialize Pinecone directly
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key:
        print("❌ ERROR: PINECONE_API_KEY not found in .env")
        return
    
    print(f"\n✓ Connecting to Pinecone index: {index_name}")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Initialize embeddings and LLM
    print("✓ Initializing embeddings and LLM...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Query
    # query = "How to do matrix multiplication in MATLAB?"
    # query = "How to do matrix division in MATLAB?"
    # query = "How to Estimate Transition Probabilities using Financial Toolbox in MATLAB?"
    # query = "How to install MATLAB?"
    # query = "How to perform matrix multiplication of C and D in Matlab if C = [1,2,3] and D = [7,8,9]? "
    
    # query = "How to do matrix multiplication of matrices A and B in Matlab ? "  # This Work
    query = "How to do matrix multiplication of matrices C and D in Matlab ? "  # Doesn't Work

    print(f"\n📝 Query: {query}")
    
    # Generate query embedding
    print("🔄 Generating query embedding...")
    query_embedding = embeddings.embed_query(query)
    
    # Search Pinecone
    print("🔍 Searching Pinecone...")
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        namespace="production"
    )
    
    # print('-'*50)

    # print('results :')
    # print(results)

   
    # Extract context from results
    context_parts = []
    for match in results.get('matches', []):
        if 'metadata' in match and 'text' in match['metadata']:
            text = match['metadata']['text']
            context_parts.append(text)
    
    context = "\n\n".join(context_parts[:3])  # Use top 3 results
    
    # print('Context : ', context)
    # print('-'*50)

    if not context:
        print("⚠ No results found in Pinecone")
        return
    
    # Create prompt and generate answer
    print("💭 Generating answer...")
#     prompt_template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know.
# Use three sentences maximum and keep the answer concise.

# CONTEXT:
# {context}

# QUESTION: {question}

# ANSWER:"""

    prompt_template = """Use the following pieces of context to answer the question at the end.
    # Answer only based on the context given.
    # If you don't know the answer, just say that you don't know.

    # CONTEXT:
    # {context}

    # QUESTION: {question}

    # ANSWER:"""

    
    prompt = PromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt.format(context=context, question=query)
    
    # Get answer from LLM
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    answer = response.content
    
    # Print results
    print("\n" + "="*70)
    print("QUERY:", query)
    print("="*70)
    print("ANSWER:")
    print(answer)
    print("="*70)
    print(f"\n✓ Successfully retrieved answer!")


if __name__ == "__main__":
    main()
