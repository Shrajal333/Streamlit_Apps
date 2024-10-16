import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
model = ChatGroq(model="Gemma2-9b-It", api_key=GROQ_API_KEY)

prompt = ChatPromptTemplate.from_template(
    """ Answer the questions based on provided context only.
        Please provide the most accurate response based on asked questions
        
        <context>
        {context}
        <context>
        
        Question:{question}
    """
)

def create_vector_embeddings():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="gemma:2b")
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title('RAG from research papers')
user_input = st.text_area('Enter the query from the research papers')

if st.button("Document Embeddigns"):
    create_vector_embeddings()
    st.write("Embeddings are ready !!!")

if user_input:
    document_chain = create_stuff_documents_chain(model, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    response = retriever_chain.invoke({"question":user_input})
    st.write(response["answer"])