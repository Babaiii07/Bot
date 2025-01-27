import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

def initialize_groq_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        max_tokens=512
    )

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.load_local(
    "medical_faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

prompt_template = """
You are a healthcare professional built by Parthib, and you can assist users with health-related issues.
Use the following pieces of information along with the LLM's knowledge to answer the user's question about diseases or healthcare.
If the following pieces provide some information, combine it with your existing knowledge to craft the most accurate and helpful response.
Include relevant details such as home remedies, medications, and other necessary actions in a clear, point-wise manner for quick readability.
If any other related questions arise, just say, "I am a healthcare professional."
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def generate_response(question):

    retriever = faiss_index.as_retriever(search_kwargs={'k': 1})
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])

    llm = initialize_groq_llm()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    formatted_prompt = prompt.format(context=context, question=question)

    response = llm.invoke(formatted_prompt)
    return response.content

st.set_page_config(page_title="HealthCare ChatBot", page_icon="🤖", layout="centered")
st.header("HealthCare ChatBot 🤖")

user_input = st.text_input("Ask a Healthcare related question:", "")
st.button("Generate Response")
st.spinner('Processing')

if user_input:
    user_input = user_input.lower().strip()
    response = generate_response(user_input)
    st.write(f"Response: {response}")
