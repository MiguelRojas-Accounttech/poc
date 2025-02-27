import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="Darwin GPT",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Modern Chat Look
st.markdown(
    """
    <style>
        body {
            background-color: #0B0F19;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #0B0F19;
        }
        .chat-container {
            background-color: #1E1E2F;
            border-radius: 10px;
            padding: 15px;
            max-height: 450px;
            overflow-y: auto;
            width: 70%;
            margin: auto;
        }
        .user-message {
            background-color: #007AFF;
            color: white;
            padding: 12px 15px;
            border-radius: 18px;
            margin-bottom: 10px;
            max-width: 75%;
            align-self: flex-end;
            text-align: left;
            display: block;
        }
        .bot-message {
            background-color: #2C2C3C;
            color: #D1D5DB;
            padding: 12px 15px;
            border-radius: 18px;
            margin-bottom: 10px;
            max-width: 75%;
            align-self: flex-start;
            text-align: left;
            display: block;
        }
        .input-container {
            position: fixed;
            bottom: 15px;
            left: 50%;
            transform: translateX(-50%);
            width: 70%;
            background-color: #0B0F19;
            padding: 10px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
        }
        .stTextInput input {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #444;
            background-color: #1E1E2F;
            color: white;
        }
        .stButton button {
            background-color: #007AFF;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #005FCC;
        }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_qa_chain():
    from pinecone import Pinecone
    import pinecone.data.index
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Pinecone as LC_Pinecone
    from langchain_community.llms import OpenAI
    from langchain.chains import RetrievalQA

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    index_name = "darwin-db"
    
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    pinecone.Index = pinecone.data.index.Index
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=openai_api_key
    )
    
    vectorstore = LC_Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

qa = load_qa_chain()

st.markdown("<h1 style='text-align: center; color: white;'>Darwin GPT</h1>", unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Chat history container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages (Newest at the bottom like ChatGPT)
for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        st.markdown(f"<div class='user-message'>**You:** {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-message'>**Bot:** {message}</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input area at the bottom
with st.container():
    query = st.text_input("Enter your query:", key="query", label_visibility="collapsed")

    col1, col2 = st.columns([0.85, 0.15])
    
    with col1:
        pass  # Empty column for alignment

    with col2:
        if st.button("Ask"):
            if query:
                with st.spinner("Querying..."):
                    answer = qa.run(query)
                
                # Store chat in session state (temporary)
                st.session_state["chat_history"].append(("You", query))
                st.session_state["chat_history"].append(("Bot", answer))
            
                # Reset input after submission
                st.rerun()
            else:
                st.error("Please enter a query.")

# Button to clear chat history
if st.button("Clear Chat History"):
    st.session_state["chat_history"] = []
    st.rerun()
