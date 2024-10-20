import os

import ollama
from tqdm import tqdm
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core import Document

from src.logger import log


def recursive_summarization(document: Document, generate_kwargs, chunk_size: int) -> str:
    """
    Recursively summarize a text using an LLM and NodeParser with TokenTextSplitter.

    :param document: Document object with text to summarize.
    :param generate_kwargs: Arguments for the model's text generation.
    :param chunk_size: Maximum token size per chunk.
    :return: Final summary of the document.
    """
    log.info("Starting recursive summarization process.")

    def recursive_summarization_helper(nodes):
        summarized_texts = []
        for node in tqdm(nodes):
            prompt = f"""### System:
            You are an expert agent in information extraction and summarization.
            ### User:
            Read the following context document:
            ---------------
            {node.get_content()}
            ---------------

            Your tasks are as follows:
            1.- Write an extensive, fluid, and continuous paragraph summarizing the most important aspects of the information you have read.
            2.- You can only synthesize your response using exclusively the information from the context document.
            ### Assistant:
            According to the context information, the summary in English is: """

            log.debug("Generated prompt for node: %s", prompt)

            try:
                if os.getenv("ENVIRONMENT") == "development":
                    response = ollama.generate(model="llama3.1:latest", prompt=prompt, **generate_kwargs)
                else:
                    client = ollama.Client(os.getenv("OLLAMA_HOST"))
                    response = client.generate(model="llama3.1:latest", prompt=prompt, **generate_kwargs)

                response_text = response['response'].strip()
                summarized_texts.append(response_text)
                log.info("Successfully generated summary for node.")

            except Exception as e:
                log.error("Error generating summary for node: %s", e)
                summarized_texts.append("Error during summarization.")

        if len(nodes) == 1:
            log.info("Returning summary for a single node.")
            return summarized_texts[0]
        else:
            combined_text = ' '.join(summarized_texts).strip().strip('\n')
            log.info("Combining summaries from multiple nodes.")
            new_nodes = node_parser.get_nodes_from_documents([Document(text=combined_text)])
            return recursive_summarization_helper(new_nodes)

    log.info("Initializing TokenTextSplitter with chunk size: %d", chunk_size)
    node_parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    initial_nodes = node_parser.get_nodes_from_documents([document])
    log.info("Splitting document into initial nodes.")

    final_summary = recursive_summarization_helper(initial_nodes)
    log.info("Completed recursive summarization process.")

    return final_summary


def get_summary(document: str) -> str:
    log.info("Generating summary for document.")
    return recursive_summarization(Document(text=document), {}, 512)
