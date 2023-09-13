from langchain.document_loaders import WikipediaLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from utils import PineconeCredentials, init_pinecone

import os
if os.path.exists("env.py"):
    import env

import pinecone

def ingest_docs(subject: str, pinecone_env: PineconeCredentials) -> None:

    pinecone.init(api_key=pinecone_env.api_key, environment=pinecone_env.environment_region)

 

    loader = WikipediaLoader(query=subject, load_max_docs=1)

    raw_text = loader.load()

 

    print(f"Loaded {subject}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(raw_text)

    

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

 

    Pinecone.from_documents(documents=documents, embedding=embeddings, index_name=pinecone_env.index_name)

 

    print("Successfully loaded vectors in Pinecone.")
 

if __name__ == "__main__":

    pinecone_env = init_pinecone()

    ingest_docs("Oppenheimer (film)", pinecone_env)
    pass