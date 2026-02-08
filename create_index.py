import os
from dotenv import load_dotenv
from streamlit import json
_ = load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import glob
from langchain_text_splitters import MarkdownHeaderTextSplitter

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
def create_vector_store_md_spliter(docs_dir='docs',save_path ="vectors/vector_store_data_md_spliter.json", embeddings=embeddings):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    docs_dir = 'docs'
    doc_splits = []

    for path in glob.glob(os.path.join(docs_dir, '*')):
        if path.lower().endswith(('.md', '.txt')):
            with open(path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
                # Use split_text, not split_documents
                header_splits = markdown_splitter.split_text(file_content)
                
                # Optional: Add the filename to metadata for each split
                for chunk in header_splits:
                    chunk.metadata["source"] = os.path.basename(path)
                
                doc_splits.extend(header_splits)
    

    vector_store = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=embeddings)

    # vector_store.save(save_path)
    vector_store.dump(save_path)
    return vector_store

def create_vector_store(docs_dir='docs',save_path ="vector_store_data.json",text_splitter=text_spliter, embeddings=embeddings):
    import glob
    from langchain.schema import Document
    docs = []
    docs_dir = 'docs'
    for path in glob.glob(os.path.join(docs_dir, '*')):
        # only load markdown and text files (adjust as needed)
        if path.lower().endswith(('.md', '.txt')):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs.append([Document(page_content=text, metadata={'source': os.path.basename(path)})])
    

    docs_list = [item for sublist in docs for item in sublist]

    doc_splits = text_splitter.split_documents(docs_list)

    

    vector_store = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=embeddings)

    # vector_store.save(save_path)
    vector_store.dump(save_path)
    return vector_store

def retrieve_context(query: str,vector_store,k=5):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

def create_index_chank_healper(vector_store_path = "vectors/vector_store_data_md_spliter.json",results_path = "vectors/vector_store_data_md_spliter_chunk.json"):
    """just a stupid helper to help me create questions on the chunks """
    #import json vector_store to list
    import json
    with open(vector_store_path, 'r', encoding='utf-8') as f:
        vector_store_data = json.load(f)
    results = []
    for k,doc in vector_store_data.items():
        doc_id = doc['id']
        content = doc['text']
        metadata = doc['metadata']
        results.append({
            "id": doc_id,
            "text": content,
            "metadata": metadata
        })
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # create_vector_store_md_spliter()
    create_index_chank_healper(vector_store_path = "vectors/vector_store_data_md_spliter.json",results_path = "vectors/vector_store_data_md_spliter_chunk.json")
