import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

parser = StrOutputParser()
model = ChatGroq(model="Gemma2-9b-It", api_key=GROQ_API_KEY)

template = 'Translate the following into {language}:'

prompt = ChatPromptTemplate.from_messages(
    [('system', template),
     ('human', '{text}')]
)

chain = prompt|model|parser

st.title('Language Translation')
user_input_language = st.text_area('Translate English to desired language')
if not user_input_language:
    user_input_language = "Japanese"

user_input_text = st.text_area('Enter the text you want to translate')

if st.button('Translate'):
    result = chain.invoke({'language':user_input_language, 'text':user_input_text})
    st.write(f'Translation: {result}')
else:
    st.write('Enter the text you want to translate')