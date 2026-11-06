from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Tyhan_Personal_RAG_Knowledge_Base.txt"
DB_PATH = BASE_DIR / "chroma_db"

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_version="v1beta"
)


def ensure_vector_store():

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Knowledge base not found: {DATA_PATH}")

    DB_PATH.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name="introduction",
        embedding_function=embedding,
        persist_directory=str(DB_PATH)
    )

    existing_count = vector_store._collection.count()

    if existing_count > 0:
        print(f"Database already contains {existing_count} documents.")
        return vector_store

    loader = TextLoader(
        str(DATA_PATH),
        encoding="utf-8"
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No document chunks were created.")

    print(f"Total chunks created: {len(chunks)}")

    vector_store.add_documents(chunks)

    print(
        f"Vector database created successfully. "
        f"Stored documents: {vector_store._collection.count()}"
    )

    return vector_store


if __name__ == "__main__":
    ensure_vector_store()