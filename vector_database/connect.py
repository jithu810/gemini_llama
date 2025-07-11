import os
import urllib3
import requests
import traceback
import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from utils.config import Config

loggers= Config.init_logging()
logger = loggers['chromadb']

def get_chroma_client(path="./chroma_db", model_name="all-MiniLM-L6-v2"):
    """
    Initializes and returns a persistent ChromaDB client along with an embedding function.
    This function sets up the ChromaDB client for persistent storage at the specified path,
    optionally disables SSL verification for HTTP requests, and initializes the embedding
    function using the specified model.
    Args:
        path (str, optional): The file system path where the ChromaDB persistent client will store data.
            Defaults to "./chroma_db".
        model_name (str, optional): The name of the sentence transformer model to use for embeddings.
            Defaults to "all-MiniLM-L6-v2".
    Returns:
        tuple: A tuple containing:
            - client: The initialized PersistentClient instance for ChromaDB.
            - embedding_func: The initialized SentenceTransformerEmbeddingFunction instance.
    Raises:
        RuntimeError: If there is an error during initialization, with detailed error information logged.
    """
    try:
        logger.info("[CHROMA-DB] Initializing ChromaDB client...")
        # Optional: Patch requests to ignore SSL
        old_request = requests.Session.request
        def unsafe_request(self, *args, **kwargs):
            kwargs['verify'] = False
            return old_request(self, *args, **kwargs)
        requests.Session.request = unsafe_request
        logger.debug("[CHROMA-DB] SSL verification disabled for requests.")
        # Create embedding function
        embedding_func = SentenceTransformerEmbeddingFunction(model_name=model_name)
        logger.info(f"[CHROMA-DB] Embedding function initialized with model: {model_name}")

        # Initialize persistent Chroma client
        client = PersistentClient(path=path)
        logger.info(f"[CHROMA-DB] ChromaDB persistent client initialized at: {path}")
        return client, embedding_func

    except Exception as e:
        err_msg = (
            f"\n[CHROMA-DB]  Error initializing ChromaDB client:\n"
            f"Type    : {type(e).__name__}\n"
            f"Message : {str(e)}\n"
            f"Trace   :\n{traceback.format_exc()}"
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg) from e
