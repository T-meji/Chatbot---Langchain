from dataclasses import dataclass
import os
if os.path.exists("env.py"):
    import env


@dataclass
class PineconeCredentials:
    api_key: str
    index_name: str
    environment_region: str


def init_pinecone():
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    environment = os.environ.get("PINECONE_ENVIRONMENT_REGION")

    if api_key is None:
        msg = "PINECONE_API_KEY environment variable is not set."
        raise ValueError(msg)
    
    if index_name is None:
        msg = "PINECONE_INDEX_NAME environment variable is not set."
        raise ValueError(msg)
    
    if environment is None:
        msg = "PINECONE_ENVIRONMENT_REGION environment variable is not set."
        raise ValueError(msg)
    
    return PineconeCredentials(
        api_key=api_key,
        index_name=index_name,
        environment_region=environment
    )