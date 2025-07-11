# -*- coding: utf-8 -*-
import os
from utils.service_validators import validate_pdf_file
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.response_utils import response as _response
from utils.prompts import SUMMARIZE_PROMPT,DEFAULT_SUMMARIZE_PROMPT
from core.document_extractor import extract_text_from_file
from utils.llama_load import load_llama_model
from interceptors.request_id_interceptor import request_id_ctx
from utils.value_check import safe_int,safe_float  
from utils.timer import Timer

from utils.config import Config
loggers = Config.init_logging()

OUTPUT_DIR= Config.OUTPUT_DIR
TEMPERATURE=Config.TEMPERATURE
MAX_NEW_TOKENS=Config.MAX_NEW_TOKENS
PRODUCTION=Config.ENVIRONMENT

service_logger = loggers['document_summary']

class DocumentSummaryProcessor:
    def __init__(self, params: dict, context):
        """
        Initializes the DocumentSummaryProcessor with parameters and context.
        :param params: Dictionary containing parameters for processing.
        :param context: Context for the service call.
        """
        service_logger.info("[INIT] DocumentSummaryProcessor initialized with parameters.")
        self.params = params
        self.context = context
        self.ROLE_SYSTEM = "system"
        self.ROLE_USER = "user"

        self.doc_path = self.params.get("FilePath")
        self.queries = self.params.get("Query")
        self.query_id = self.params.get("QueryId")
        self.raw_text = self.params.get("RawText")
        
        self.temperature = safe_float(self.params.get("temperature", TEMPERATURE), TEMPERATURE)
        self.max_new_tokens = safe_int(self.params.get("max_new_tokens", MAX_NEW_TOKENS), MAX_NEW_TOKENS)

        if not self.query_id:   
            service_logger.warning(f"[INIT] Missing QueryId parameter.")
        request_id_ctx.set(self.query_id)
        
        self.model,error=load_llama_model()
        if error:
            message= ErrorMessages.MODEL_LOAD_FAILED
            service_logger.error(f"[INIT MODEL FAILED] {message} {error}")
            raise RuntimeError(f"{message}: {error}")

    def process(self):
        """
        Processes a document for summarization using a language model.
        This method performs the following steps:
        1. Logs the start of the process with relevant parameters.
        2. Validates the presence and type of the document file.
        3. Extracts text from the provided document file.
        4. Prepares the prompt and message list for the summarization model.
        5. Streams the summary from the model, accumulating the result.
        6. Handles and logs errors at each step, returning appropriate HTTP responses.
        Returns:
            dict: A response dictionary containing status code, description, remarks, and summarized data,
                  or an error response if any step fails.
        """
        try:
            service_logger.info(f"[START] QueryId={self.query_id}, FilePath={self.doc_path}, Temperature={self.temperature},"
                                f"MaxTokens={self.max_new_tokens},RawText={self.raw_text}")
            if not self.doc_path:
                message = ErrorMessages.MISSING_PARMS
                error_message = f"{message} PDF path not provided."
                service_logger.warning(f"[INIT]{error_message}")
                return _response(HttpStatusCodes.BAD_REQUEST, message, error_message)

            is_valid, file_type_or_error = validate_pdf_file(self.doc_path)
            if not is_valid:
                message= ErrorMessages.ERROR_VAL_DOC
                error_message = f"[NOT VALID]{message} {file_type_or_error}"
                service_logger.warning(error_message)
                return _response(HttpStatusCodes.BAD_REQUEST,message,error_message)
            try:
                input_text = extract_text_from_file(self.doc_path,file_type_or_error)
                if not input_text or input_text.strip() == "":
                    raise ValueError("Extracted text is empty.")
            except Exception as e:
                message = ErrorMessages.AI_EXTRACTION_FAILED
                error_message= f"{message} {str(e)}"
                service_logger.error(error_message)
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,message,error_message)

            # Determine user message
            user_message = self.queries[0] if isinstance(self.queries, list) and self.queries else DEFAULT_SUMMARIZE_PROMPT
            message_list = [
                {"role": self.ROLE_SYSTEM, "content": f"{SUMMARIZE_PROMPT}:\n\n{input_text}"},
                {"role": self.ROLE_USER, "content": user_message}
            ]

            service_logger.info("Calling the LlamaSummarizer for document summarization...")
            summary_accumulator = ""

            with Timer() as total_timer:
                for result in self.model.stream_summary(max_new_tokens=self.max_new_tokens, temperature=self.temperature, messages=message_list,query_id=self.query_id,use_history=False):
                    if isinstance(result, dict) and "error" in result:
                        message = ErrorMessages.SUMMARY_FAILED
                        error_message= f"{message} {result['error']}"
                        service_logger.error(error_message)
                        return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,message,error_message)
                    else:
                        if PRODUCTION !='production':
                            print(result,end="",flush=True)
                            # service_logger.info(f"Streamed chunk: {result.strip()}")
                        summary_accumulator += result
            result = {
                "QueryId": self.query_id,
                "Time": f"{total_timer.interval:.2f}s",
                "Data": summary_accumulator
            }
            service_logger.info(f"[DATA SENT]{result}")
            return {
                "status_code": HttpStatusCodes.OK,
                "status_description": "OK",
                "remarks": SuccessMessages.PROCESSED_SUCCESSFULLY,
                "data": result
            }

        except Exception as e:
            message= ErrorMessages.INTERNAL_SERVER_ERROR
            error_message= f"[INTERNAL SERVER ERROR]{message}: {str(e)}"
            service_logger.error(error_message)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,message,error_message)