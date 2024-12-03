import streamlit as st
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

# Validate API Key
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

def main():
    st.title("Groq Chat App")

    # Sidebar options
    st.sidebar.title('Settings')
    model = st.sidebar.selectbox(
        'Choose an LLM model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096']
    )
    conversational_memory_length = st.sidebar.slider(
        'Conversational memory length (number of messages):', 
        min_value=1, 
        max_value=10, 
        value=5
    )

    # Initialize memory for conversation
    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load chat history into memory
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Initialize Groq chat object
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    # Initialize conversation chain
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    # User input field
    user_question = st.text_area("Ask a question:")

    # Handle user input and generate response
    if user_question:
        response = conversation.run(user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
