import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from src.indexing_layer import ensure_vector_store
from src.user_layer import create_rag_chain

logging.basicConfig(level=logging.INFO)

log = logging.getLogger("twin")


@asynccontextmanager
async def lifespan(app: FastAPI):

    log.info("Preparing ChromaDB...")

    ensure_vector_store()

    log.info("Initializing Tyhan AI RAG chain...")

    app.state.chain = create_rag_chain()

    log.info("RAG chain initialized successfully.")

    yield

    log.info("Shutting down Tyhan AI.")


app = FastAPI(
    title="Tyhan AI API",
    description="Personal AI Assistant powered by RAG",
    version="1.0.0",
    lifespan=lifespan
)


class Ask(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=500
    )

    conversation_id: str | None = None


def to_text(result):

    if isinstance(result, str):
        return result

    if isinstance(result, dict):

        for key in ("answer", "result", "output", "text"):

            if key in result:
                return str(result[key])

    return str(getattr(result, "content", result))


@app.get("/api/health")
async def health():

    return {
        "status": "running",
        "assistant": "Tyhan AI"
    }


@app.post("/api/chat")
async def chat(req: Ask):

    try:

        chain = app.state.chain

        result = await run_in_threadpool(
            chain.invoke,
            req.question.strip()
        )

        return {
            "answer": to_text(result),
            "sources": []
        }

    except Exception:

        log.exception("RAG chain failed")

        raise HTTPException(
            status_code=502,
            detail="The assistant is temporarily unavailable."
        )


# Serve frontend LAST
app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend"
)