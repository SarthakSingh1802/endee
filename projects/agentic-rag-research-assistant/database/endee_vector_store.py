import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EndeeVectorStore:
    """
    Vector store implementation using Endee database principles.
    For simplicity, uses in-memory storage with cosine similarity search.
    In a real implementation, this would interface with the C++ Endee database.
    """
    def __init__(self):
        """
        Initialize the vector store.
        """
        self.embeddings = []  # List of numpy arrays
        self.documents = []   # List of document strings

    def add_documents(self, embeddings, documents):
        """
        Add document embeddings to the store.
        :param embeddings: List of embedding vectors (numpy arrays).
        :param documents: List of document contents.
        """
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)

    def similarity_search(self, query_embedding, top_k=5):
        """
        Perform similarity search on the stored embeddings.
        :param query_embedding: Embedding vector for the query.
        :param top_k: Number of top similar documents to return.
        :return: List of top-k similar documents.
        """
        if not self.embeddings:
            return []
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]