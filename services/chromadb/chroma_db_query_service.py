# -*- coding: utf-8 -*-

from utils.config import Config
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.response_utils import response as _response
from utils.timer import Timer
from interceptors.request_id_interceptor import request_id_ctx

from vector_database.connect import get_chroma_client
loggers= Config.init_logging()
service_logger = loggers['chromadb']

class ChromaQueryProcessor:
    def __init__(self, params: dict, context):
        """
        Initializes the ChromaQueryProcessor with parameters and context.
        :param params: Dictionary containing parameters for processing.
        :param context: Context for the service call.
        """
        service_logger.info("[INIT] ChromaQueryProcessor initialized with parameters.")
        self.params = params
        self.context = context

        self.query_id = self.params.get("QueryId")
        self.source_filter = self.params.get("Source")
        self.query = self.params.get("Query")
        self.query_type = self.params.get("QueryType", "").lower()

        try:
            self.max_results = int(self.params.get("MaxResults", 5))
        except (ValueError, TypeError):
            self.max_results = 5

        if not self.query_id:
            service_logger.warning(f"[INIT] Missing QueryId parameter.")
            raise ValueError("Missing QueryId parameter.")
        request_id_ctx.set(self.query_id)

        try:
            self.client, self.embedding_func = get_chroma_client()
        except RuntimeError as err:
            service_logger.error(f"ChromaDB initialization failed: {err}")
            self.client = None
            self.embedding_func = None

    def process(self):
        """
        Processes a query request to the ChromaDB service, handling both 'list' and 'query' operations.
        This method performs the following steps:
        1. Logs the start of the process with query details.
        2. Validates that the ChromaDB client, query type, and embedding function are initialized.
        3. Retrieves or creates the 'uploaded_documents' collection using the provided embedding function.
        4. Depending on the query type:
            - 'list': Retrieves documents and metadata, optionally filtered by source.
            - 'query': Performs a similarity search using the provided query text and returns matching documents.
        5. Logs the results and execution time.
        6. Handles and logs errors, returning appropriate HTTP status codes and messages.
        Returns:
            dict: A response dictionary containing status code, description, remarks, data (documents, metadatas, ids, etc.), 
                  or an error response if validation or processing fails.
        """
        try:
            service_logger.info(
                f"[START] QueryId={self.query_id}, QueryType={self.query_type}, MaxResults={self.max_results}")
            if self.client is None:
                messages= ErrorMessages.CLIENT_NOT_INITIALIZED
                error_message = f"{messages} ChromaDB client not initialized."
                service_logger.warning(error_message)
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,messages,error_message)
            if not self.query_type:
                messages= ErrorMessages.MISSING_PARMS
                error_message= f"{messages} QueryType parameter is missing or empty."
                service_logger.warning(error_message)
                return _response(HttpStatusCodes.BAD_REQUEST, messages, error_message)
            if self.embedding_func is None:
                messages = ErrorMessages.EMBEDDING_FUNC_NOT_INITIALIZED
                error_message = f"{messages} Embedding function not initialized."
                service_logger.warning(error_message)
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, messages, error_message)
            
            collection = self.client.get_or_create_collection(
                name="uploaded_documents",
                embedding_function=self.embedding_func
            )
            service_logger.info(f"Processing query type: {self.query_type}")
            
            if self.query_type == "list":
                with Timer() as timer:
                    if self.source_filter:
                        results = collection.get(
                            where={"source": self.source_filter},
                            include=["metadatas", "documents"]
                        )
                    else:
                        results = collection.get(include=["metadatas", "documents"])

                documents = results.get("documents", [])[:self.max_results]
                metadatas = results.get("metadatas", [])[:self.max_results]
                ids = results.get("ids", [])[:self.max_results]
                service_logger.info(f"Retrieved {len(documents)} documents in {timer.interval:.2f}s")

                return {
                    "status_code": HttpStatusCodes.OK,
                    "status_description": "OK",
                    "remarks": SuccessMessages.PROCESSED_SUCCESSFULLY,
                    "data": {
                        "documents": documents,
                        "metadatas": metadatas,
                        "ids": ids,
                        "count": len(documents),
                        "Time": f"{timer.interval:.2f}s"
                    }
                }

            elif self.query_type == "query":
                with Timer() as query_timer:
                    query_args = {
                        "query_texts": [self.query],
                        "n_results": self.max_results,
                    }
                    if self.source_filter:
                        query_args["where"] = {"source": self.source_filter}
                    results = collection.query(**query_args)

                ids = results.get("ids", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                documents = results.get("documents", [[]])[0]
                distances = results.get("distances", [[]])[0]
                service_logger.info(f"Query returned {len(documents)} results in {query_timer.interval:.2f}s")
                return {
                    "status_code": HttpStatusCodes.OK,
                    "status_description": "OK",
                    "remarks": SuccessMessages.PROCESSED_SUCCESSFULLY,
                    "data": {
                        "ids": ids,
                        "metadatas": metadatas,
                        "documents": documents,
                        "distances": distances,
                        "count": len(documents),
                        "Time": f"{query_timer.interval:.2f}s"
                    }
                }
            else:
                service_logger.warning(f"Invalid QueryType: {self.query_type}")
                return _response(HttpStatusCodes.BAD_REQUEST, "Invalid or missing QueryType parameter")

        except Exception as e:
            messages= ErrorMessages.INTERNAL_SERVER_ERROR
            error_message = f"{messages}: {str(e)}"
            service_logger.error(error_message)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,messages,error_message)
