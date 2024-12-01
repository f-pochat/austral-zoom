from fastapi import UploadFile
from torch import Tensor
from src import speech_to_text, summary, embeddings


def process_audio_service(audio: UploadFile) -> (str, str, list[tuple[str, Tensor]]):
    audio_text = speech_to_text.process(audio)
    summary_text = summary.get_summary(audio_text)

    embeddings_vector = embeddings.get_embeddings(audio_text)
    return audio_text, summary_text, embeddings_vector
