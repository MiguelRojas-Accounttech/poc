import os
from dotenv import load_dotenv
import streamlit as st

@st.cache_resource  # or st.experimental_singleton in older Streamlit versions
def load_qa_chain():
    # Load environment variables and configure API
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    index_name = "darwin-db"  # Verify the name is correct

    # Set OpenAI key in environment
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Import and patch Pinecone
    from pinecone import Pinecone
    import pinecone.data.index
    pinecone.Index = pinecone.data.index.Index

    # Create Pinecone instance and get index
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    # Initialize embeddings model
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=openai_api_key
    )

    # Connect LangChain vectorstore to Pinecone index
    from langchain_community.vectorstores import Pinecone as LC_Pinecone
    vectorstore = LC_Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    # Create question answering (QA) chain using OpenAI as LLM
    from langchain_community.llms import OpenAI
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

# Load QA chain (this operation is performed only once)
qa = load_qa_chain()

# Streamlit interface
st.title("Q&A Interface with Pinecone and GPT")

query = st.text_input("Enter your query:")

if st.button("Ask"):
    if query:
        with st.spinner("Querying..."):
            answer = qa.run(query)
        st.markdown("**Answer:**")
        st.write(answer)
    else:
        st.error("Please enter a query.")
