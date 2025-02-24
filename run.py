import os
from dotenv import load_dotenv
from pinecone import Pinecone  # No se usa pinecone.init

# Importa la vectorstore y el LLM desde langchain-community
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_community.llms import OpenAI

# Importa el modelo de embeddings (puedes usar langchain_openai)
from langchain_openai import OpenAIEmbeddings

# Importa la cadena de QA
from langchain.chains import RetrievalQA

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Recupera las variables de entorno
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# La variable pinecone_env ya no es necesaria aquí
openai_api_key = os.getenv("OPENAI_API_KEY")
index_name = "darwin-db"  # Asegúrate de usar solo minúsculas y guiones si es necesario

# Establece la clave de OpenAI en el entorno
os.environ["OPENAI_API_KEY"] = openai_api_key

# Crea una instancia de la clase Pinecone con la API key
pc = Pinecone(api_key=pinecone_api_key)
# Obtén el índice existente
index = pc.Index(index_name)

# Inicializa el modelo de embeddings (por ejemplo, text-embedding-ada-002)
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=openai_api_key
)

# Conecta la vectorstore de LangChain al índice de Pinecone
vectorstore = LC_Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Crea la cadena de preguntas y respuestas (QA) utilizando OpenAI como LLM
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",  # "stuff" concatena los documentos recuperados
    retriever=vectorstore.as_retriever()
)

print("Conexión establecida con Pinecone. Puedes comenzar a preguntar.")
while True:
    query = input("Ingresa tu consulta (o 'exit' para terminar): ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = qa.run(query)
    print("Respuesta:", answer)
