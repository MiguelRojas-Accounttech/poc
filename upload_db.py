import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import CharacterTextSplitter

# Try to import the new OpenAIEmbeddings from langchain_community;
# if not available, you might need to install/update langchain-community.
try:
    from langchain_community.embeddings import OpenAIEmbeddings
except ImportError:
    from langchain_openai import OpenAIEmbeddings

# Load environment variables from the .env file
load_dotenv()

# Instantiate the new Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "darwin-db"

# Create the index if it does not exist
existing_indexes = [idx.name for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,       # Make sure this matches your embedding model's dimension (e.g., 1536 for text-embedding-ada-002)
        metric="cosine",      # Use "cosine" (or "euclidean", "dotproduct" as needed)
        spec=ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_ENV")  # e.g., "us-east-1"
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Read the text content from the file "data.txt"
with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split the text into chunks using LangChain's text splitter
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0)
chunks = text_splitter.split_text(text)

# Initialize the embeddings model (using text-embedding-ada-002, for example)
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Prepare the list of vectors to upsert
vectors = []
for i, chunk in enumerate(chunks):
    # Generate embedding for the chunk
    vector_values = embeddings.embed_query(chunk)
    # Create a unique ID and include the chunk text as metadata
    vector = (f"chunk-{i}", vector_values, {"text": chunk})
    vectors.append(vector)

# Upsert the vectors into Pinecone if there are any
if vectors:
    index.upsert(vectors)
    print(f"{len(vectors)} chunks uploaded to Pinecone successfully!")
else:
    print("No chunks to upload.")
