from embeddings.embedding_model import EmbeddingModel
from database.endee_vector_store import EndeeVectorStore

class RetrievalAgent:
    """
    Agent responsible for retrieving relevant documents based on the query.
    """
    def __init__(self, embedding_model, vector_store):
        """
        Initialize the retrieval agent.
        :param embedding_model: Instance of EmbeddingModel.
        :param vector_store: Instance of EndeeVectorStore.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k relevant documents for the query.
        :param query: Processed query string.
        :param top_k: Number of documents to retrieve.
        :return: List of relevant documents.
        """
        query_embedding = self.embedding_model.generate_embedding(query)
        return self.vector_store.similarity_search(query_embedding, top_k)