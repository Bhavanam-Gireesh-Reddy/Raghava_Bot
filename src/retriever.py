class mainretriever:
    def __init__(self):
        pass
    def Topkretriever(self,question, db, TOP_K):
        retriever = db.as_retriever(search_kwargs = {"k": TOP_K})
        retrived_docs = retriever.invoke(question)

        context="\n\n".join(doc.page_content for doc in retrived_docs)
        return context
    
    def mmr_retrival(self, question, db, TOP_K):
        retriever = db.max_marginal_relevance_search(query = question, k = TOP_K, fetch_k = 20, lambda_mult = 0.5)
        retrived_docs = retriever.invoke(question)

        context="\n\n".join(doc.page_content for doc in retrived_docs)
        return context