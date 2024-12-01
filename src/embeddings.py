import ollama
import torch
from torch import Tensor

from src.logger import log

from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings


def get_embeddings(audio_text: str) -> list[tuple[str, Tensor]]:
    log.info("Generating embeddings for audio text")
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    chunks = text_splitter.split_text(audio_text)
    items = []
    for chunk in chunks:
        embedding = generate_embeddings(chunk)
        items.append((chunk, embedding))
    return items


def generate_embeddings(chunk: str) -> torch.Tensor:
    return ollama.embeddings(model="tinyllama", prompt=chunk)["embedding"]
