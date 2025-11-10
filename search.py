import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from src.ddg_fallback import DDGWebFallback
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain.agents import create_agent
# from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient


load_dotenv()
# os.environ["TAVILY_API_KEY"] = 
# print(response)
class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.tavily_client = TavilyClient(api_key="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")
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = "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"
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

        self.web_fallbacker = DDGWebFallback(top_k=3)

    def web_search(self, query):
        response = self.tavily_client.search(query)
        return response['results']
    
    # def search_and_summarize(self, query: str, top_k: int = 5) -> str:
    #     #Query local vectorstore
    #     results = self.vectorstore.query(query, top_k=top_k)
    #     texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
    #     context = "\n\n".join(texts)
    #     if not context:
    #         return "Sorry I have no data on your question. Ask me anything related to MongoDB."
    #     prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
    #     response = self.llm.invoke([prompt])
    #     return response.content

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Search local vectorstore first. If no data, fallback to DuckDuckGo web search. """
        # 1️⃣ Query local vectorstore
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)

        if context:
            # Local data found -> summarize with LLM
            prompt = f"Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"
            response = self.llm.invoke([prompt])
            return response.content
    
        else:
            web_results = self.web_search(query=query)
            content = ""
            for each in web_results:
                content+=f"title is {each['title']} and content is {each['content']}"
            # print("web_results : ", web_results)
            prompt = f"Summarize the following context for the query: '{query}'\n\nContext:\n{content}\n\nSummary:"
            response = self.llm.invoke([prompt])
            return response.content
            return web_results

        # return self.web_fallbacker.fallback(query, llm=self.llm)