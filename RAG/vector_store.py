import os
import faiss
import numpy as np
import pickle
from typing import List,Any
from RAG.embedding import EmbeddingPipeline
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class FaissVectorStore:
    def __init__(self,persist_dir: str="faiss_store",embedding_model:str = "all-MiniLM-L6-V2",chunk_size:int=1000,chunk_overlap:int=100):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir,exist_ok=True)
        self.index = None
        self.metadata = []
        self.documents = []
        self.embedding_model = embedding_model
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunk_size=chunk_size
        self.chunk_overlap = chunk_overlap
        self.bm25_retriever = None
        print(f"Loaded embedding model : {embedding_model}")

    def build_from_documents(self,documents:List[Any]):
        print(f"Building vector store from {len(documents)} documents")
        chunks = self.embedding_pipeline.chunk_documents(documents)
        self.documents = chunks
        self.bm25_retriever = BM25Retriever.from_documents(
            chunks,
            k=10
        )
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        metadatas = [{"text":chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'),metadatas)
        self.save()
        print(f"Vector store built successfully")

    def add_embeddings(self,embeddings:np.ndarray,metadata:List[Any]) -> None:
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadata:
            self.metadata.extend(metadata)
        print(f"Added {embeddings.shape[0]} vectors to Faiss index")

    def save(self):
        faiss_path = os.path.join(self.persist_dir,"faiss.index")
        meta_path = os.path.join(self.persist_dir,"metadata.pk1")
        faiss.write_index(self.index,faiss_path)
        with open(meta_path,"wb") as f:
            pickle.dump(self.metadata,f)
        print(f"Saved faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir,"faiss.index")
        meta_path = os.path.join(self.persist_dir,"metadata.pk1")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path,"rb") as f:
            self.metadata = pickle.load(f)
        self.documents = [
            Document(page_content=meta["text"])
            for meta in self.metadata
        ]
        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            k=10
        )
        print(f"Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self,query_embeddings:np.ndarray,top_k:int=5):
        D,I = self.index.search(query_embeddings,top_k)
        results=[]
        for idx,dist in zip(I[0],D[0]):
            meta = self.metadata[idx] if idx<len(self.metadata) else None
            results.append({"index":idx,"distance":dist,"metadata":meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"Querying Vector store for {query_text}")
        query_emb = self.embedding_pipeline.model.encode(query_text).astype("float32")
        query_emb = np.expand_dims(query_emb, axis=0)

        return self.search(query_emb, top_k)

    def hybrid_search(self, query_text: str, top_k: int = 5):

        faiss_results = self.query(query_text,top_k=10)
        bm25_results = self.bm25_retriever.invoke(query_text)
        scores = {}

        # FAISS ranking
        for rank, result in enumerate(faiss_results):
            text = result["metadata"]["text"]
            scores[text] = scores.get(text, 0) + (1 / (60 + rank + 1))

        # BM25 ranking
        for rank, document in enumerate(bm25_results):
            text = document.page_content
            scores[text] = scores.get(text, 0) + (1 / (60 + rank + 1))

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [text for text, score in ranked[:top_k]]