try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import sys
sys.path.insert(0, '.')
from utils.document_loader import DocumentLoader
from embeddings.embedding_model import EmbeddingModel
from database.endee_vector_store import EndeeVectorStore
from agents.query_agent import QueryAgent
from agents.retrieval_agent import RetrievalAgent
from agents.reasoning_agent import ReasoningAgent

# Initialize components
embedding_model = EmbeddingModel()
vector_store = EndeeVectorStore()
query_agent = QueryAgent()
retrieval_agent = RetrievalAgent(embedding_model, vector_store)
reasoning_agent = ReasoningAgent(embedding_model)

# Load documents from data/research_papers
loader = DocumentLoader('data/research_papers')
documents = loader.load_documents()

# Generate embeddings and store in vector store
if documents:
    embeddings = embedding_model.generate_embeddings(documents)
    vector_store.add_documents(embeddings, documents)

if STREAMLIT_AVAILABLE:
    # Streamlit UI
    st.title("Agentic AI Research Assistant using RAG with Endee Vector Database")

    st.markdown("Ask research questions and get answers based on retrieved documents.")

    query = st.text_input("Enter your research question:")

    if st.button("Get Answer"):
        if query:
            # Process query
            processed_query = query_agent.process_query(query)
            
            # Retrieve relevant documents
            retrieved_docs = retrieval_agent.retrieve(processed_query, top_k=3)
            
            # Generate answer
            answer = reasoning_agent.generate_answer(query, retrieved_docs)
            
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Retrieved Documents (Top 3):")
            for i, doc in enumerate(retrieved_docs, 1):
                st.write(f"**Document {i}:** {doc[:200]}...")
        else:
            st.warning("Please enter a question.")
else:
    # CLI mode
    if len(sys.argv) > 1:
        # Command line query
        query = sys.argv[1]
        processed_query = query_agent.process_query(query)
        retrieved_docs = retrieval_agent.retrieve(processed_query, top_k=3)
        answer = reasoning_agent.generate_answer(query, retrieved_docs)
        print("Question:", query)
        print("Answer:", answer)
        print("\nRetrieved Documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"Document {i}: {doc[:200]}...")
    else:
        print("Agentic AI Research Assistant (CLI Mode)")
        print("========================================")
        if documents:
            print(f"Loaded {len(documents)} documents.")
            while True:
                query = input("Ask a research question (or 'quit' to exit): ")
                if query.lower() == 'quit':
                    break
                processed_query = query_agent.process_query(query)
                retrieved_docs = retrieval_agent.retrieve(processed_query, top_k=3)
                answer = reasoning_agent.generate_answer(query, retrieved_docs)
                print("\nAnswer:", answer)
                print("\nRetrieved Documents:")
                for i, doc in enumerate(retrieved_docs, 1):
                    print(f"Document {i}: {doc[:200]}...")
                print("-" * 50)
        else:
            print("No documents found in data/research_papers/")