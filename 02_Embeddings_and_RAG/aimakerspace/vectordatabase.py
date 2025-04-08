import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Optional, Any, Union
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = {}  # Store metadata for each key
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Insert a vector with optional metadata into the database.
        
        Args:
            key: The key to identify the vector
            vector: The vector to insert
            metadata: Optional metadata to associate with the vector
        """
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        include_metadata: bool = False,
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """
        Search for the top k vectors closest to the query vector.
        
        Args:
            query_vector: The query vector
            k: Number of results to return
            distance_measure: Function to measure distance between vectors
            include_metadata: Whether to include metadata in the results
            
        Returns:
            List of tuples containing (key, score) or (key, score, metadata)
        """
        if include_metadata:
            scores = [
                (key, distance_measure(query_vector, vector), self.metadata.get(key))
                for key, vector in self.vectors.items()
            ]
        else:
            scores = [
                (key, distance_measure(query_vector, vector))
                for key, vector in self.vectors.items()
            ]
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        include_metadata: bool = False,
    ) -> Union[List[Tuple[str, float]], List[str], List[Tuple[str, float, Optional[Dict[str, Any]]]]]:
        """
        Search for the top k text items closest to the query text.
        
        Args:
            query_text: The query text
            k: Number of results to return
            distance_measure: Function to measure distance between vectors
            return_as_text: Whether to return only the text
            include_metadata: Whether to include metadata in the results
            
        Returns:
            List of tuples containing (key, score) or (key, score, metadata), or just list of texts
        """
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, include_metadata)
        
        if return_as_text:
            return [result[0] for result in results]
        return results

    def retrieve_from_key(self, key: str, include_metadata: bool = False) -> Union[np.array, Tuple[np.array, Dict[str, Any]]]:
        """
        Retrieve a vector by its key, optionally including metadata.
        
        Args:
            key: The key to retrieve
            include_metadata: Whether to include metadata in the result
            
        Returns:
            The vector or a tuple of (vector, metadata)
        """
        vector = self.vectors.get(key, None)
        if vector is None:
            return None
            
        if include_metadata:
            return (vector, self.metadata.get(key))
        return vector

    async def abuild_from_list(
        self, 
        list_of_text: List[str], 
        list_of_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> "VectorDatabase":
        """
        Build the database asynchronously from a list of texts with optional metadata.
        
        Args:
            list_of_text: List of texts to embed
            list_of_metadata: Optional list of metadata for each text
            
        Returns:
            Self for chaining
        """
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        
        if list_of_metadata:
            for text, embedding, metadata in zip(list_of_text, embeddings, list_of_metadata):
                self.insert(text, np.array(embedding), metadata)
        else:
            for text, embedding in zip(list_of_text, embeddings):
                self.insert(text, np.array(embedding))
                
        return self
        
    async def abuild_from_list_with_metadata(
        self, 
        texts_with_metadata: List[Tuple[str, Dict[str, Any]]]
    ) -> "VectorDatabase":
        """
        Build the database asynchronously from a list of (text, metadata) tuples.
        
        Args:
            texts_with_metadata: List of (text, metadata) tuples
            
        Returns:
            Self for chaining
        """
        list_of_text = [text for text, _ in texts_with_metadata]
        list_of_metadata = [metadata for _, metadata in texts_with_metadata]
        
        return await self.abuild_from_list(list_of_text, list_of_metadata)


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
