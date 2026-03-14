from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    """
    Embedding model using Sentence Transformers.
    Uses the 'all-MiniLM-L6-v2' model for generating embeddings.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.
        :param model_name: Name of the sentence transformer model.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text):
        """
        Generate embedding for a given text.
        :param text: Input text string.
        :return: Numpy array of the embedding vector.
        """
        return self.model.encode(text)

    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        :param texts: List of input text strings.
        :return: Numpy array of embedding vectors.
        """
        return self.model.encode(texts)