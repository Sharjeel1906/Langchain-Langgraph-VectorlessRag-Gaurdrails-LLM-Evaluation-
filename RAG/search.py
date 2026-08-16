import os
from dotenv import load_dotenv
from RAG.vector_store import FaissVectorStore
from langchain_groq import ChatGroq
from RAG.data_loader import load_all_documents
from RAG.re_ranking import DocumentReranker
load_dotenv()

class RAGSearch:
    def __init__(self,persist_dir:str="faiss_store",embedding_model:str="all-MiniLM-L6-V2"):
        self.vector_store = FaissVectorStore(persist_dir,embedding_model)
        self.reranker = DocumentReranker()
        # load and built vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pk1")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_all_documents("data")
            self.vector_store.build_from_documents(docs)
        else:
            self.vector_store.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(api_key=groq_api_key,model_name="llama-3.3-70b-versatile")
        print(f"Groq LLM Initialized")

    def search_and_summerize(self,query:str,top_k:int=3):
        #results = self.vector_store.query(query,top_k=10)
        #texts = [r["metadata"].get("text") for r in results if r["metadata"]]
        texts = self.vector_store.hybrid_search(query, top_k=10)

        print(texts)

        re_ranked_texts = self.reranker.rerank(query_text=query,top_k=top_k,documents=texts)
        print(re_ranked_texts)

        context = "\n\n".join(re_ranked_texts)
        if not context:
            return "No relevant documents found"
        prompt = f""" 
        Use the following context to answer the question concisely.
       Context:{context}
       Question:{query}
       """
        response =  self.llm.invoke([prompt])
        return response.content
