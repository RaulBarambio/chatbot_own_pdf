import pandas as pd
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):

    ''' Coge los texto que extrae del pdf '''

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text

def get_text_chunks(text):

    ''' Dividir el texto en varios fragmentos para poder acotar la respuesta'''

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len 
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):

    '''Genera los embeddings y los almacena en FAISS'''

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def add_data_db_faiss(local_path, documents, status):

    '''Esta función hace que si el status es new te genera una bbdd y si es add te añade a tu bbdd original datos nuevos'''
    embeddings = OpenAIEmbeddings()

    if status == 'new':
        db_new = FAISS.from_documents(documents, embeddings)
        db_new.save_local(local_path)

    elif status == 'add':
        vector_DB = FAISS.load_local(local_path, embeddings)
        db_new = FAISS.from_documents(documents, embeddings)
        vector_DB.merge_from(db_new)
        vector_DB.save_local(local_path)

def create_faiss_index(path, name_index):
    #Importar los archivos que deseas agregar a la bbdd
    data = get_pdf_text(path)

    # Divides el texto en chunks, lo puedes ajustar en la funcion
    text_chunks = get_text_chunks(data)

    #Genera la bbdd y la gaurda en la ruta data/nombre que le pongas
    db = get_vectorstore(text_chunks)
    db.save_local(f"data/{name_index}")


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPEN_API_KEY")
    openai.organization = os.getenv("OPENAI_ORGANIZATION_ID")
    embeddings = OpenAIEmbeddings()
    path = r'C:\Users\rbarambi\OneDrive - NTT DATA EMEAL\Escritorio\seguros_procesados\cg-autoplus-29-03-2023.pdf'

    #Importar los archivos que deseas agregar a la bbdd y la guardas
    create_faiss_index(path, 'bbdd_faiss')

    #Cargas la bbdd en la path que se encuentre y con los embeddings de OpenAI en este caso
    db_faiss = FAISS.load_local(local_path, embedding=embeddings)

if __name__ == '__main__':
    main()
