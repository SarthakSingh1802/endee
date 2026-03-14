# Agentic AI Research Assistant using RAG with Endee Vector Database

This project implements an AI-powered research assistant that uses Retrieval-Augmented Generation (RAG) with the Endee vector database to answer research questions based on uploaded documents.

## Features

- Document loading from text files
- Embedding generation using Sentence Transformers
- Vector storage and similarity search with Endee
- Agentic workflow: Query processing, Retrieval, and Reasoning
- Streamlit web interface for user interaction

## Tech Stack

- Python
- Sentence Transformers (all-MiniLM-L6-v2)
- Endee Vector Database
- LangChain-inspired agent architecture
- Streamlit for UI
- NumPy, Pandas, scikit-learn

## Project Structure

```
agentic-rag-research-assistant/
├── data/
│   └── research_papers/  # Place your .txt research documents here
├── agents/
│   ├── query_agent.py      # Processes user queries
│   ├── retrieval_agent.py  # Retrieves relevant documents
│   └── reasoning_agent.py  # Generates answers from context
├── database/
│   └── endee_vector_store.py  # Vector storage interface
├── embeddings/
│   └── embedding_model.py   # Embedding generation
├── utils/
│   └── document_loader.py   # Document loading utility
├── app/
│   └── main.py             # Streamlit application
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Place your research documents (.txt files) in the `data/research_papers/` folder.

3. Run the Streamlit app:
   ```
   streamlit run app/main.py
   ```

## Usage

1. Start the Streamlit app.
2. Enter a research question in the text input.
3. Click "Get Answer" to retrieve relevant documents and generate a response.

## Components

- **EmbeddingModel**: Generates vector embeddings for texts.
- **DocumentLoader**: Loads text documents from a folder.
- **EndeeVectorStore**: Stores embeddings and performs similarity search.
- **QueryAgent**: Preprocesses user queries.
- **RetrievalAgent**: Converts queries to embeddings and retrieves documents.
- **ReasoningAgent**: Combines context to generate answers (placeholder for LLM integration).

## Future Enhancements

- Integrate with actual LLM (e.g., OpenAI GPT) for better answer generation.
- Support for PDF and other document formats.
- Persistent vector storage with Endee database.
- Advanced agent workflows with LangChain.