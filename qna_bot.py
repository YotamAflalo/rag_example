from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import os
from dotenv import load_dotenv
_ = load_dotenv()
model = init_chat_model("gpt-4.1")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def retrieve_context(query: str,vector_store,k=5,eval=False):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    if eval:
        ids = [doc.id for doc in retrieved_docs]
        if 'Unknown ID' in ids:
            print("Warning: Some retrieved documents are missing 'ids' in metadata.")

    return {"serialized": serialized, "ids": ids if eval else []}
def load_vector_store(file_path):
    # Load your retriever from the file
    # This is a placeholder implementation, replace it with your actual loading logic
    vector_store = InMemoryVectorStore.load(file_path, embedding=embeddings)
    return vector_store

def answer_query(query: str, vector_store, model,eval = False):
    retrived = retrieve_context(query, vector_store,eval = eval)
    context = retrived["serialized"]
    if eval: ids = retrived["ids"]
    results = model.invoke([
    {"role": "system", "content": f"""Use the following context to answer the question:{query}. 
     If the context does not contain the answer, say you don't know. if you use spesific information from the context, 
     tell the user the source headers."""},
    {"role": "user", "content": f"Context:\n{context}\n\n Question: {query}? /n/n your answer: /n/n"}
    ]).content
    if eval:
        return {"results": results, "context": context, "ids": ids}
    return {"results": results}

if __name__ == "__main__":
    vector_store = load_vector_store("vectors/vector_store_data_md_spliter.json")
    query = " "
    while query.lower() != "exit":
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() != "exit":
            answer = answer_query(query, vector_store, model)['results']
            print(f"Answer: {answer}\n")
    # query = "What is the purpose of this project?"
    # answer = answer_query(query, vector_store, model)
    # print(answer)