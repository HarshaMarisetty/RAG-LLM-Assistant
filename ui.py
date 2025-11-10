import streamlit as st
import time
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
# from openai import OpenAI

st.title("Mongo-Bot")
rag_search = RAGSearch()
#docs = load_all_documents("data")
store = FaissVectorStore("faiss_store")
#store.build_from_documents(docs)
store.load()
# Set OpenAI API key from Streamlit secrets
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    summary = rag_search.search_and_summarize(prompt, top_k=3)

    # Display user message in chat message container
    with st.chat_message("user"):
        # print(prompt, "Hellowww")
        st.markdown(prompt)

 # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        def live_stream():
            for word in summary.split(" "):
                yield word + " "
                time.sleep(0.09)
            # ["", summary]
        response = st.write_stream(live_stream)
    st.session_state.messages.append({"role": "assistant", "content": response})




# Example usage
# if __name__ == "__main__":
    

    #print(store.query("What is Replication?", top_k=3))

    # query = "Describe about the data present in the documents?"
    # print("Summary:", summary)#llama-3.1-8b-instants