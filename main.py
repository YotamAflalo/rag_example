#!/usr/bin/env python
import streamlit as st
from qna_bot import answer_query, load_vector_store
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
loaded = load_dotenv()
# Initialize model, embeddings and vector store (reused from existing setup)
model = init_chat_model("gpt-4.1")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = load_vector_store("vectors/vector_store_data_md_spliter.json")


def main():
	st.set_page_config(page_title="RAG Chat", page_icon="🤖")
	st.title("RAG Chat — Ask the Docs")

	if "history" not in st.session_state:
		st.session_state.history = []

	with st.sidebar:
		st.header("Settings")
		# k = st.number_input("Retrieve top k documents", min_value=1, max_value=20, value=5)
		if st.button("Clear chat"):
			st.session_state.history = []

	query = st.text_input("Enter your question:")
	col1, col2 = st.columns([4,1])
	with col2:
		send = st.button("Send")

	if send and query:
		st.session_state.history.append(("user", query))
		with st.spinner("Generating answer..."):
			answer = answer_query(query, vector_store, model).get('results', "Sorry, I couldn't generate an answer.")
		st.session_state.history.append(("bot", answer))

	# Display chat history
	for role, message in st.session_state.history:
		if role == "user":
			st.markdown(f"**You:** {message}")
		else:
			st.markdown(f"**Bot:** {message}")


if __name__ == "__main__":
	main()

