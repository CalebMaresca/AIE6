import os
from typing import List, Dict, Tuple


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.metadata = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())
            # Extract the filename without extension as document name
            doc_name = os.path.basename(self.path)
            self.metadata.append({"name": doc_name})

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(
                        file_path, "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())
                        # Extract the filename without extension as document name
                        self.metadata.append({"name": file})

    def load_documents(self):
        self.load()
        if self.metadata:
            return list(zip(self.documents, self.metadata))
        else:
            return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        """
        Split texts while preserving metadata for each chunk.
        
        Args:
            texts_with_metadata: List of (text, metadata) tuples
            
        Returns:
            List of (chunk, metadata) tuples
        """
        chunks_with_metadata = []
        for text, metadata in texts:
            # Split the text into chunks
            text_chunks = self.split(text)
            # Create a (chunk, metadata) tuple for each chunk
            for chunk in text_chunks:
                chunks_with_metadata.append((chunk, metadata))
        return chunks_with_metadata


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
