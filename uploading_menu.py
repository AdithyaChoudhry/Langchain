import streamlit as st
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone as PineconeVector
from dotenv import load_dotenv
import random
import time
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain_community.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
# LLama index stuff
import os
import pinecone
from pinecone import PodSpec

#from llama_index.vector_stores import PineconeVectorStore
#from llama_index.storage.storage_context import StorageContext
#from llama_index.embeddings import GeminiEmbedding
#from llama_index import ServiceContext, VectorStoreIndex, download_loader, set_global_service_context
from langchain.docstore.document import Document


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def delete_all_vectors():
    index = pc.Index("agrikiosk")
    index.delete(delete_all=True, namespace="")


def get_pdf_doc(pdf_docs):
    text=""
    doc = []
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    doc.append(Document(page_content=text, metadata={"source": "local"}))
    return doc

def get_doc_chunks(doc):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
    chunks = text_splitter.split_documents(doc)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index = pc.Index("agrikiosk")
    index_name = "agrikiosk"
    vector_store = PineconeVector(index,embeddings,"text_content")
    batch_size = 100  # Define your preferred batch size
    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i:i + batch_size]
        st.write(chunk_batch)
        vector_store.add_documents(chunk_batch)
    print("Ok")   
    return vector_store 

def get_conversation_chain(vectorstore):
    model= ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,convert_system_message_to_human=True)
    # llm = ChatOpenAI(temperature=0.3, model_name="gpt-4-turbo-preview")
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})


    
        # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all details, if the answer is not in provided context just say, "Answer is not available in the context". don't provide the wrong answer
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])

    qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
    )

    tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the topic'
        )
    )
    ]

    agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=model,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
    )

    return agent
    


def main(): 
    #delete_all_vectors()
    st.set_page_config("SSNGPT DEMO")
    st.header("SSNGPT: Chat with GPT 🤖")

    if "disabled" not in st.session_state:
        st.session_state.disabled=True

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    with st.sidebar:
        st.title("Menu:")
        if st.button('Delete Knowledge Base'):
            delete_all_vectors()
            st.session_state.disabled=True
        pdf_docs = st.file_uploader("Upload your PDF Files for Course and Click on the Submit button",accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_doc(pdf_docs)
                text_chunks = get_doc_chunks(raw_text)
                vectorstore = get_vector_store(text_chunks)
                st.success("Done")
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.disabled=False
                #print(st.session_state.conversation)


    if prompt := st.chat_input("What is up?",disabled=st.session_state.disabled):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = st.session_state.conversation(prompt)['output']
                st.write(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":

    main()
