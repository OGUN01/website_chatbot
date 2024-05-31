#from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface.llms import HuggingFaceEndpoint
import os


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


load_dotenv()


llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3")


# Create a new Chroma client

#llm = Ollama(model="llama3")



embeddings = HuggingFaceEmbeddings()





import streamlit as st


st.set_page_config(page_title="chat with website",page_icon="book")

st.title("chat with website")

def get_vector_store(url):
    documents = WebBaseLoader(url).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(docs,embedding=embeddings)
    return vector_store
     
def get_context_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([

        
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","""Given the above conversation, generate a search query to look up in order to get information relevant to the conversation and one thing only return the queary nothing else""")
         ])
    
    retriever_chain =create_history_aware_retriever(llm = llm, retriever=retriever,prompt=prompt)

    return retriever_chain
    

def get_conversational_rag_chain(retriever_chain): 
    
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

with st.sidebar:
    st.header("settings")
    website_url = st.text_input("enter your webstie url")


if website_url is None or website_url == "":
    st.info("please enter your webstie url")

else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="hi i am a robot how can i help you")]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store(website_url)

    retrivel_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retrivel_chain)




    user_queary = st.chat_input("enter your questions")




    if user_queary is not None and user_queary!="":
        

        response = get_response(user_queary)    

        st.session_state.chat_history.append(HumanMessage(content=user_queary))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)   
        

    with st.sidebar:
        st.write(st.session_state.chat_history)
        
        