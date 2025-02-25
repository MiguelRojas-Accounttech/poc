import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="Q&A Interface with Pinecone and GPT",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Apply custom styles
st.markdown(
    """
    <style>
        body {
            background-color: #174376;
            color: white;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background-color: #174376;
            padding: 20px;
            border-radius: 10px;
        }
        .stTextInput input {
            background-color: transaparent;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton button {
            background-color: transaparent;
            color: white;
            border-radius: 5px;
            font-size: 16px;
        }
        .stMarkdown {
            background-color: transaparent;
            color: white;
            padding: 15px;
            border-radius: 8px;
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

st.markdown("<h1 style='text-align: center; color: white;'>Q&A Interface with Pinecone and GPT</h1>", unsafe_allow_html=True)

query = st.text_input("Enter your query:", key="query")

if st.button("Ask"):
    if query:
        with st.spinner("Querying..."):
            answer = qa.run(query)
        st.markdown("**Answer:**", unsafe_allow_html=True)
        st.markdown(f"<div class='stMarkdown'>{answer}</div>", unsafe_allow_html=True)
    else:
        st.error("Please enter a query.")
