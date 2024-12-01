from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_text_splitters import CharacterTextSplitter

from src.logger import log

model = OllamaLLM(model="tinyllama", language="es")


def get_summary(document: str) -> str:
    log.info(f"Getting summary for document")

    prompt = """
                      Por favor, escribir un resumen del siguiente texto. En español.
                      TEXTO: {text}
                      RESUMEN:
                      """

    question_prompt = PromptTemplate(
        template=prompt, input_variables=["text"]
    )

    refine_prompt_template = """
                  Escribe un resumen conciso en español del siguiente texto, delimitado por tres comillas invertidas.

                  {texto}
                    
                  Devuelve tu respuesta en viñetas, cubriendo los puntos clave del texto.
                    
                  RESUMEN EN VIÑETAS:
                  """

    refine_prompt = PromptTemplate(
        template=refine_prompt_template, input_variables=["text"]
    )

    # Load refine chain
    chain = load_summarize_chain(
        llm=model,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents([Document(page_content=document)])
    res = chain({"input_documents": split_docs}, return_only_outputs=True)
    return res["output_text"]
