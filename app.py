import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPEN_API_KEY")


def get_pdf_text(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

def get_vector(text_chunks, index_name="replace with your index name"):
    """Generate vectors from text chunks."""
    embedings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore.from_texts(
        text_chunks, embedings, index_name=index_name
    )
    return vector_store

def get_conversation_chain(vectorstore):
    """Initialize conversation chain."""
    model = ChatOpenAI(api_key=openai_api_key, temperature=0.2)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=model, memory=memory, retriever=vectorstore.as_retriever())
    return conversation_chain

def handle_user_question(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    is_user_message = True
    for message in st.session_state.chat_history:
        if is_user_message:
            st.markdown(f'<div style="background-color: #C0C0C0; border-radius: 10px; padding: 10px; margin: 5px 0;"><b>User:</b> {message.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: #d3d3d3; border-radius: 10px; padding: 10px; margin: 5px 0;"><b>Chatbot:</b> {message.content}</div>', unsafe_allow_html=True)
        is_user_message = not is_user_message
def main():
    """Main function."""
    # Set page configuration
    st.set_page_config("Chat with multiple txt files")
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    with st.sidebar:
        # File uploader for uploading text files
        txt_docs = st.file_uploader("Upload text files", type="txt", accept_multiple_files=True)
        # Button to process uploaded files
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Process uploaded text files
                raw_text = ""
                for txt in txt_docs:
                    raw_text += txt.getvalue().decode("utf-8")
                chunks = get_text_chunks(raw_text)
                vector_store = get_vector(chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)
    
    # Text input for user to enter question
    user_question = st.text_input("Enter your question")
    if user_question:
        handle_user_question(user_question)

if __name__ == "__main__":
    main()
