import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ReasoningAgent:
    """
    Agent to reason over retrieved documents and generate an answer.
    Extracts the most relevant sentences from the top document based on query similarity.
    """
    def __init__(self, embedding_model):
        """
        Initialize with the embedding model.
        :param embedding_model: Instance of EmbeddingModel.
        """
        self.embedding_model = embedding_model

    def generate_answer(self, query, retrieved_docs):
        """
        Generate an answer by finding the most relevant sentences in the top retrieved document.
        :param query: The original query.
        :param retrieved_docs: List of retrieved document strings.
        :return: Concatenated relevant sentences or a message if none.
        """
        if not retrieved_docs:
            return "No relevant documents found for the query: " + query
        
        # Take the top document
        doc = retrieved_docs[0]
        
        # Split into sentences (simple split by . ? !)
        sentences = re.split(r'(?<=[.!?])\s+', doc.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return doc  # Fallback to full doc
        
        # Generate embeddings for sentences
        sentence_embeddings = self.embedding_model.generate_embeddings(sentences)
        
        # Generate embedding for query
        query_embedding = self.embedding_model.generate_embedding(query)
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], sentence_embeddings)[0]
        
        # Get top 3 most similar sentences
        top_indices = np.argsort(similarities)[-3:][::-1]
        relevant_sentences = [sentences[i] for i in top_indices if similarities[i] > 0.1]  # Threshold to filter low similarity
        
        if relevant_sentences:
            return " ".join(relevant_sentences)
        else:
            return doc  # Fallback