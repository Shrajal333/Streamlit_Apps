import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma-7b-It", api_key=api_key)

import validators
import streamlit as st
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

template = """
                Provide a detailed summary of the following content 
                Content: {text}
           """
prompt = PromptTemplate(template=template, input_variables=['text'])

st.title("Summarize text from any URL including Youtube videos")
st.subheader("Summarize URL")

url = st.text_input("URL", label_visibility="collapsed")

if st.button("Start summarizing"):
    if not url.strip():
        st.error("Please enter the URL to get started")
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(url, ssl_verify=False)

            data = loader.load()
            chain = load_summarize_chain(model, chain_type='stuff', prompt=prompt)

            summary = chain.run(data)
            st.success(summary)

        except Exception as e:
            st.exception(f"Exception: {e}")