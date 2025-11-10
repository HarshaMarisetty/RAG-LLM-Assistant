# ddg_fallback.py
"""
DuckDuckGo HTML Search fallback for RAGSearch.
Fetches web pages when vectorstore has no answer and optionally summarizes them using LLM.
"""

import time, requests
from bs4 import BeautifulSoup

class DDGWebFallback:
    def __init__(self, top_k=3):
        self.top_k = top_k
        self.headers = {"User-Agent": "my-chatbot/1.0 (+https://example.com/contact)"}

    def search(self, query: str):
        """
        Perform DuckDuckGo HTML search and return top K URLs.
        """
        try:
            start=time.time()
            resp = requests.post(
                "https://html.duckduckgo.com/html",
                data={"q": query},
                headers=self.headers,
                timeout=8
            )

            print("\n\n\33[92m Total time taken : ", time.time()-start, "\33[0m ")
            # print("resp : ", resp, resp.text)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            print("\n\n\33[92m soup : ", soup, "\33[0m ")

            anchors = soup.select("a.result__a")
            print("anchors: ", anchors)
            anchors = anchors[:self.top_k]
            urls = [a.get("href") for a in anchors if a.get("href")]
            return urls
        except:
            return []

    def fetch_page_snippet(self, url: str, max_chars=2000):
        """
        Fetch a webpage, extract main text or first paragraphs, return as snippet.
        """
        try:
            time.sleep(0.3)  # polite delay
            resp = requests.get(url, headers=self.headers, timeout=8)
            print("resp : ", resp)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # remove unnecessary tags
            for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
                tag.decompose()
            # prefer <main> or <article>
            main = soup.find("main") or soup.find("article")
            paras = main.find_all("p") if main else soup.find_all("p")[:5]
            text = " ".join([p.get_text(" ", strip=True) for p in paras])
            return text[:max_chars]
        except Exception as e:
            print("Error over exception :", e)
            return ""

    def fallback(self, query: str, llm=None):
        """
        Search web, fetch snippets, summarize using LLM if provided.
        Returns summary string.
        """
        urls = self.search(query)
        snippets = [self.fetch_page_snippet(url) for url in urls if url]
        snippets = [s for s in snippets if s]
        print("snippets: ", snippets)
        if not snippets:
            return "Sorry, I couldn't find relevant information on the web."

        context = "\n\n".join(snippets)
        if llm:
            prompt = f"Summarize the following web search results for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"
            response = llm.invoke([prompt])
            return response.content
        else:
            # fallback: return raw text if no LLM
            return context[:1000]
