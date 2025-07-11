# -*- coding: utf-8 -*-
import os
# from utils.config import loggers, TEMPERATURE, MAX_NEW_TOKENS
from utils.config import Config
from utils.prompts import BASE_RAG_SYSTEM_PROMPT
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.timer import Timer
from utils.response_utils import response as _response
from utils.llama_load import load_llama_model
from vector_database.connect import get_chroma_client
from interceptors.request_id_interceptor import request_id_ctx
from utils.value_check import safe_int,safe_float
from utils.history import load_history,save_history

loggers = Config.init_logging()

TEMPERATURE=Config.TEMPERATURE
MAX_NEW_TOKENS=Config.MAX_NEW_TOKENS
PRODUCTION=Config.ENVIRONMENT

service_logger = loggers['chatservice']

class ChatProcessor:
    def __init__(self, params: dict, context):
        """
        Initializes the ChatProcessor with parameters and context.
        :param params: Dictionary containing parameters for processing.
        :param context: Context for the service call.
        """
        service_logger.info("[INIT] ChatProcessor initialized with parameters.")
        self.params = params
        self.context = context
        self.ROLE_SYSTEM = "system"
        self.ROLE_USER = "user"
        self.ROLE_ASSISTANT = "assistant"

        self.source = self.params.get("Source")
        self.clearchat = self.params.get("clearchat")
        self.queries = self.params.get("Query")
        self.query_id = self.params.get("QueryId")

        self.temperature = safe_float(self.params.get("temperature", TEMPERATURE), TEMPERATURE)
        self.max_new_tokens = safe_int(self.params.get("max_new_tokens", MAX_NEW_TOKENS), MAX_NEW_TOKENS)
    
        if not self.query_id:   
            service_logger.warning(f"[INIT] Missing QueryId parameter.")
        request_id_ctx.set(self.query_id)
        service_logger.info(f"[INIT] QueryId={self.query_id}, Source={self.source}, "
                            f"Temp={self.temperature}, MaxTokens={self.max_new_tokens}, queries={self.queries}")
        try:
            self.client, self.embedding_func = get_chroma_client()
            service_logger.info("[CHROMA] Client and embedding function initialized successfully.")
        except RuntimeError as err:
            service_logger.error(f"[CHROMA ERROR] {err}")
            self.client = None
            self.embedding_func = None

        self.model,model_error = load_llama_model()
        if model_error:
            service_logger.error(f"[MODEL FAILED] {model_error}")
            self.model = None
        else:
            service_logger.info("[MODEL] Model loaded successfully.")

    def fetch_context_from_chroma(self, query_text):
        if self.client is None:
            service_logger.error("[CHROMA] Client not initialized.")
            return None, "ChromaDB client not initialized"

        if self.embedding_func is None:
            service_logger.error("[CHROMA] Embedding function not initialized.")
            return None, "Embedding function not initialized"

        try:
            collection = self.client.get_or_create_collection(
                name="uploaded_documents", embedding_function=self.embedding_func
            )
            query_args = {
                "query_texts": [query_text],
                "n_results": 3,
            }
            service_logger.info(f"[CHROMA] Query: {query_text} | Source: {self.source}")
            if self.source:
                query_args["where"] = {"source": self.source}
                service_logger.info(f"[CHROMA] Applying filter: {query_args['where']}")

            results = collection.query(**query_args)
            documents = results.get("documents", [[]])[0]
            if not documents:
                service_logger.warning("[CHROMA] No documents found for context.")
                return None, "No context documents found"
            service_logger.info(f"[CHROMA] Retrieved {len(documents)} documents for context.")
            return "\n".join(documents), None

        except Exception as e:
            service_logger.error(f"[CHROMA QUERY ERROR] {e}")
            return None, f"Error querying ChromaDB: {str(e)}"

    def clear_chat(self):
        history_path = f"./history/{self.query_id}.json"
        try:
            if os.path.exists(history_path):
                os.remove(history_path)
                service_logger.info(f"[CLEAR CHAT] Deleted history for QueryId={self.query_id}")
            else:
                service_logger.info(f"[CLEAR CHAT] No history file found for QueryId={self.query_id}")
        except Exception as e:
            service_logger.error(f"[CLEAR CHAT ERROR] {e}")
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, f"Error clearing chat: {str(e)}")

        return {
            "status_code": HttpStatusCodes.OK,
            "status_description": "OK",
            "remarks": "Chat history cleared successfully",
            "data": {
                "QueryId": self.query_id,
                "Response": "Chat history cleared.",
                "Time": "0.00s"
            }
        }

    def process(self):
        service_logger.info(
                f"[PROCESS] QueryId={self.query_id}, Source={self.source}, "
                f"Temp={self.temperature}, MaxTokens={self.max_new_tokens}, queries={self.queries}")
        
        if str(self.clearchat).lower() == "true":
            service_logger.info(f"[PROCESS] Clear chat request received for QueryId={self.query_id}")
            return self.clear_chat()
        
        if not self.model:
            message= ErrorMessages.MODEL_LOAD_FAILED
            error_message = f"[MODEL FAILED]{message} Model not loaded."
            service_logger.warning(f"[PROCESS] {error_message}")
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, message, error_message)

        if not self.client:
            message = ErrorMessages.MODEL_LOAD_FAILED
            error_message = f"[CLIENT ERROR] {message} Chroma client not initialized."
            service_logger.warning(error_message)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,message,error_message)
        
        if not self.queries or not self.query_id:
            message = ErrorMessages.MISSING_PARMS
            error_message = f"[MISSING PARAMETERS] {message} QueryId or queries not provided."
            service_logger.warning(error_message)
            return _response(HttpStatusCodes.BAD_REQUEST,message,error_message)

        history, history_error = load_history(self.query_id)
        if history_error:
            message= ErrorMessages.HISTORY_LOAD_FAILED
            history_error = f"{message} {history_error}"
            service_logger.warning(history_error)    
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, message, history_error)
        
        if not isinstance(history, list):
            message = ErrorMessages.HISTORY_LOAD_FAILED
            error_message = f"[HISTORY LOAD ERROR] Expected list, got {type(history)}"
            service_logger.warning(error_message)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, message, error_message)
        
        context_chunk, context_error = self.fetch_context_from_chroma(self.queries[0])
        if context_error:
            message = ErrorMessages.CONTEXT_FETCH_FAILED
            context_error = f"{message} {context_error}"
            service_logger.warning(context_error)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,message, context_error)
        
        system_prompt = BASE_RAG_SYSTEM_PROMPT.format(context=context_chunk)
        if not system_prompt:
            message = ErrorMessages.SYSTEM_PROMPT_NOT_FOUND
            error_message = f"[SYSTEM PROMPT ERROR] {message} No system prompt found."
            service_logger.warning(error_message)
            return _response(HttpStatusCodes.BAD_REQUEST,error_message, error_message)
        
        full_message_list = [{"role": self.ROLE_SYSTEM, "content": system_prompt}] + history
        full_message_list.append({"role": self.ROLE_USER, "content": self.queries[0]})

        service_logger.info(f"[START CHAT] QueryId={self.query_id}, messages={len(full_message_list)}")
        response_text = ""
        try:
            with Timer() as total_timer:
                for chunk in self.model.stream_summary(
                    messages=full_message_list,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                ):
                    response_text += chunk
                    if PRODUCTION !='production':
                        print(chunk, end="", flush=True)

            full_message_list.append({"role": self.ROLE_ASSISTANT, "content": response_text})
            success, error = save_history(full_message_list[-6:],self.query_id)
            if not success:
                message = ErrorMessages.HISTORY_SAVE_FAILED
                error = f"{message} {error}"
                service_logger.error(error)
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, message, error)
            else:
                service_logger.info(f"[HISTORY SAVED] {len(full_message_list[-6:])} messages saved for QueryId={self.query_id}")
            if not response_text:
                message = ErrorMessages.EMPTY_RESPONSE
                error_message = f"[EMPTY RESPONSE] {message} No response generated."
                service_logger.warning(error_message)
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, message, error_message)
            service_logger.info(f"[DATA SENT] QueryId={self.query_id}, Response Length={len(response_text)}")
            # Prepare the final response
            return {
                "status_code": HttpStatusCodes.OK,
                "status_description": "OK",
                "remarks": SuccessMessages.PROCESSED_SUCCESSFULLY,
                "data": {
                    "QueryId": self.query_id,
                    "Response": response_text,
                    "Time": f"{total_timer.interval:.2f}s"
                }
            }
        
        except Exception as e:
            message = ErrorMessages.INTERNAL_SERVER_ERROR
            error_message = f"[PROCESS ERROR] {message}: {str(e)}"
            service_logger.error(error_message)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, message, error_message)

