import ssl
import uvicorn
from fastapi import FastAPI, UploadFile
import logging
from src import service, logger

ssl._create_default_https_context = ssl._create_unverified_context
logger.init_loggers()

app = FastAPI()


@app.post("/")
def process_audio(file: UploadFile):
    return service.process_audio_service(audio=file)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
