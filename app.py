import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import chromadb.api
chromadb.api.client.SharedSystemClient.clear_system_cache()
from dotenv import load_dotenv


load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## set up streamlit app
st.title("Conversational Chatbot with speech support")

# Initialize the language model
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='gemma2-9b-it')

# chat interface 
session_id = st.text_input("Session ID : Enter a new session id to start a new conversation without chat-history", value="Default")

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

if 'store' not in st.session_state:
    st.session_state.store = {}

# Load and split documents
loader = TextLoader('My_data.txt')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(data)

# Create embeddings and vector database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(documents=splits, embedding=st.session_state.embeddings)
retriver = vector_db.as_retriever()

## prompt to read store history and create a new prompt with history 
contextulize_qa_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed, otherwise return it as is."
)
contextulize_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextulize_qa_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', "{input}")
    ]
)  

history_aware_retriver = create_history_aware_retriever(llm, retriver, contextulize_qa_prompt) 

## Answer question
system_prompt = (
    """You are an AI answering as kunal in an interview.
        Based on the context provided about me, respond to the question asked 
        in a concise, natural, and authentic way, as if you were me. 
        Use a professional yet friendly tone. 
        Do NOT make up information that is not in the context provided.Say you don't know
        if you don't have enough information. 
        AGAIN , DO NOT MAKE UP INFORMATION THAT IS NOT IN THE CONTEXT PROVIDED.
        to give a believable answer."""
            "\n\n"
            "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriver, question_answer_chain)

conversationnal_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


# def text_to_speech(text):
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
#             tts = gTTS(text=text, lang='en')  # 'en' for English
#             tts.save(tmpfile.name)
#             return tmpfile.name
#     except Exception as e:
#         return None
    

# user_input = st.text_input('What would you like to know about me ?')
# if user_input:
#     session_history = get_session_history(session_id)
#     response = conversationnal_rag_chain.invoke(
#         {'input': user_input},
#         config={
#             "configurable": {"session_id": session_id}
#         }
#     )
#     st.write(response['answer'])  # Display the answer on the web app

#     with st.spinner("Generating speech..."):
#         audio_file_path = text_to_speech(response['answer'])
#         if audio_file_path:
#             st.audio(audio_file_path, format='audio/mp3')
#         else:
#             st.warning("Failed to generate speech.")


from gtts import gTTS
import tempfile
import base64
import os
import streamlit as st
from streamlit.components.v1 import html
import time

def text_to_speech(text):
    """
    Convert text to speech using gTTS and return the audio data as bytes.
    """
    try:
        # Create a temporary MP3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
            tts = gTTS(text=text, lang='en')  # 'en' for English
            tts.save(tmpfile.name)
            tmpfile_path = tmpfile.name
        
        # Read the audio data into memory
        with open(tmpfile_path, 'rb') as f:
            audio_data = f.read()
        
        # Clean up the temporary file
        os.remove(tmpfile_path)
        return audio_data
    except Exception as e:
        return None

# Main Streamlit app
user_input = st.text_input('What would you like to know about me?')
if user_input:
    session_history = get_session_history(session_id)
    response = conversationnal_rag_chain.invoke(
        {'input': user_input},
        config={
            "configurable": {"session_id": session_id}
        }
    )
    st.write(response['answer'])  # Display the answer on the web app

    with st.spinner("Generating speech..."):
        audio_data = text_to_speech(response['answer'])
        if audio_data:
            # Encode audio data to base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            # Create a unique ID for each audio element
            unique_id = str(int(time.time() * 1000))
            # Create HTML with audio and script to ensure playback
            audio_html = f'''
            <audio id="audio_{unique_id}" style="display:none;">
              <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            <script>
              document.getElementById("audio_{unique_id}").play();
            </script>
            '''
            # Render the HTML to play the audio automatically
            html(audio_html, height=0)
        else:
            st.warning("Failed to generate speech.")
