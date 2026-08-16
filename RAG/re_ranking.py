from sentence_transformers import CrossEncoder


class DocumentReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query_text: str, documents, top_k: int = 3):

        pairs = [[query_text, doc] for doc in documents]

        scores = self.model.predict(pairs)

        ranked_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, score in ranked_docs[:top_k]]