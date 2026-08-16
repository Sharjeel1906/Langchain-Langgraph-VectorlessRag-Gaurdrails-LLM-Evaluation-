import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List,Any

class EmbeddingPipeline:
    """Handles document embeddings generation using sentence-transformers"""
    def __init__(self,model_name:str="all-MiniLM-L6-V2",chunk_size:int=1000,chunk_overlap:int=100):

        """
        Initializes Embedding Manager
        Args:
            model_name:HuggingFace model name for sentence embeddings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Loaded {model_name} model")

    def chunk_documents(self,documents:List[Any])->List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n","\n"," ","","."]
        )
        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def embed_chunks(self,chunks:List[Any])->np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"Generating Embedding for {len(chunks)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Embedding shape: {embeddings.shape}")
        return embeddings