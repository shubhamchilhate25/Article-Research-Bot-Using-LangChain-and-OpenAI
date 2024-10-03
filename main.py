import os
import streamlit as st
import pickle
import time
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Importing required libraries for retry and error handling
import openai
import requests
from requests.exceptions import Timeout

# Load environment variables
load_dotenv()

# Set up the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Missing API key! Please set it in the environment variables.")

# Streamlit Title and Sidebar
st.title("Article Research BOT🤖")
st.sidebar.title("Article URLs♨️")

# Collect URLs
urls = []
for i in range(3):
    urls.append(st.sidebar.text_input(f"URL {i+1}"))

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placefolder = st.empty()

# Initialize OpenAI LLM with environment-provided API key
llm = OpenAI(openai_api_key=api_key, temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Ensure URLs are provided
    if not any(urls):
        st.warning("Please provide at least one URL.")
    else:
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=[url for url in urls if url])
        main_placefolder.text("Data Loading... Started... 🔃🔃")
        data = loader.load()

        # Split data into chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ' '],
            chunk_size=700
        )
        main_placefolder.text("Text Splitter... Started... ⛏️🔃")
        docs = text_splitter.split_documents(data)

        vectorstore_openai = None  # Initialize variable

        # Create embeddings using OpenAI Embeddings with error handling
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placefolder.text("Embedding Vector Started Building... 🔃🔃")
            time.sleep(2)

        except Timeout:
            st.error("The request to OpenAI's embedding API timed out. Please try again later.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred with the embedding API: {e}")
        except Exception as e:
            st.error(f"Unexpected error occurred: {e}")

        # Only attempt to save if vectorstore_openai was created successfully
        if vectorstore_openai is not None:
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)
            main_placefolder.text("FAISS index successfully saved! ✅")
        else:
            st.error("Failed to create embeddings. FAISS index not saved.")

# Handle the query input and retrieval process
query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                # Display the answer
                st.header("Answer")
                st.write(result["answer"])

                # Display sources
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources: ")
                    sources_list = sources.split("\n")  # Split sources by newline
                    for source in sources_list:
                        st.write(source)
        except EOFError:
            st.error("The FAISS index file is empty or corrupted. Please process the URLs again.")
        except Exception as e:
            st.error(f"An error occurred while loading the vector store: {e}")
    else:
        st.warning("Vector store not found. Please process the URLs first.")
