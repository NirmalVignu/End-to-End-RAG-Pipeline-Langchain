

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
import os
from langchain import hub
from dotenv import load_dotenv
from PyPDF2 import PdfReader
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')


st.set_page_config(
    page_title="RAG Application",
    page_icon=":orange_heart:",
)
st.title("RAG Application")
st.session_state.tools=[]

def restart_assistant():
    
    st.session_state["auto_rag_assistant"] = None
    st.session_state["auto_rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()


def main() -> None:
    
    # Get LLM model
    llm_model = st.sidebar.selectbox("Select LLM", options=["llama3-8b-8192", "gemma-7b-it"])
    huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)
    # Set assistant_type in session state
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    print(llm_model)
    input_url = st.sidebar.text_input(
            "Add URL to Knowledge Base", type="default", 
            #key=st.session_state["url_scrape_key"]
        )
    add_url_button = st.sidebar.button("Add URL")
    if add_url_button:
        if input_url is not None:
            alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
            loader1=WebBaseLoader(input_url)
            docs=loader1.load()
            documnets=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
            vectorstore1=FAISS.from_documents(documnets,huggingface_embeddings)
            retriver1=vectorstore1.as_retriever()
            retriever_tool_url=create_retriever_tool(retriver1,"website_search",
                      "Search for information about the question asked in the website search tool, if you find anything you can use that information or else you can search in another tools")
            st.session_state.tools.append(retriever_tool_url)
            
            
            

        # Add PDFs to knowledge base
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 100

    uploaded_file = st.sidebar.file_uploader(
            "Add a PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
        )
    if uploaded_file is not None:
        alert = st.sidebar.info("Processing PDF...", icon="🧠")
        auto_rag_name = uploaded_file.name.split(".")[0]
        # save the file temporarily
        #tmp_location = os.path.join('/tmp', uploaded_file.name)
        #loader2=PyPDFLoader([uploaded_file.read().decode()])
        pdf_reader = PdfReader(uploaded_file)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()
        #docs2=loader2.load()
        documents2=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_text(text)
        vectorstore2=FAISS.from_texts(documents2,huggingface_embeddings)
        retriver12=vectorstore2.as_retriever()
        retriever_tool_pdf=create_retriever_tool(retriver12,"pdf_search",
                      "Search for information about the question asked in the pdf search tool, if you find anything you can use that information or else you can search in another tools")
        st.session_state.tools.append(retriever_tool_pdf)
        

   

    print(len(st.session_state.tools))

    if st.sidebar.button("New Run"):
        restart_assistant()

    prompt1=st.text_input("Ask your question ?")
    
    if prompt1:
        llm=ChatGroq(temperature=0.3,model_name=llm_model)
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = initialize_agent(llm=llm,
                         tools=st.session_state.tools,
                         prompt=prompt,
                         agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                         verbose=True,
                         output_key = "result",
                         handle_parsing_errors = True,
                         max_iterations=3,
                         early_stopping_method="generate",
                         memory = ConversationBufferMemory(memory_key = 'chat_history')                    
                        )
        response=agent.invoke({"input":prompt1})
        st.write(response['output'])

        



main()