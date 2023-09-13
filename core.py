from typing import Dict
from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from utils import PineconeCredentials, init_pinecone
import os

import pinecone

def run_llm(pinecone_env: PineconeCredentials) -> Dict:

    pinecone.init(api_key=pinecone_env.api_key, environment=pinecone_env.environment_region)

 

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

 

    doc_search = Pinecone.from_existing_index(

        index_name=pinecone_env.index_name,

        embedding=embeddings

    )

 

    chat_model = ChatOpenAI(

        model_name="gpt-3.5-turbo",

        temperature=1,

        verbose=True

    )

 

    qa = ConversationalRetrievalChain.from_llm(

        llm=chat_model,

        retriever=doc_search.as_retriever()

    )

 

    chat_history = []

 

    print("Start a conversation!")

    while True:

        query = input("You:")

        if query == "quit":

            break

        response = qa({"question": query, "chat_history": chat_history}).get("answer")

 

        print(f"Response: {response}")

        chat_history.append(response)

if __name__ == "__main__":

    pinecone_env = init_pinecone()

    run_llm(pinecone_env)