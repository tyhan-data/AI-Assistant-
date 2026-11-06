from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter


# Load environment variables
load_dotenv()


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Tyhan_Personal_RAG_Knowledge_Base.txt"

DB_PATH = BASE_DIR / "chroma_db"


# Initialize embedding model
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# Create vector store
def create_vector_store():

    # Load documents
    loader = TextLoader(str(DATA_PATH), encoding="utf-8")
    docs = loader.load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    # Store documents in Chroma
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name="introduction",
        persist_directory=str(DB_PATH)
    )

    print("Vector database created successfully!")

    return vector_store


# Run indexing only when this file is executed directly
if __name__ == "__main__":
    create_vector_store()