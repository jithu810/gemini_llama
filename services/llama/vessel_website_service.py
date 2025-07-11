# -*- coding: utf-8 -*-
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.timer import Timer
from utils.response_utils import response as _response
from utils.prompts import WEBSITE_PROMPT,WEBSITE_PROMPT2
from utils.llama_load import load_llama_model
from interceptors.request_id_interceptor import request_id_ctx
from utils.value_check import safe_int,safe_float
from utils.config import Config
from langchain_community.document_loaders import WebBaseLoader

import re
loggers = Config.init_logging()

OUTPUT_DIR= Config.OUTPUT_DIR
TEMPERATURE=Config.TEMPERATURE
MAX_NEW_TOKENS=Config.MAX_NEW_TOKENS
PRODUCTION=Config.ENVIRONMENT

service_logger = loggers['document_summary']

class VesselWebsiteProcessor:
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

        self.query_id = self.params.get("QueryId")
        self.url = self.params.get("url")  
        
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
        
    def simple_clean_text(self,text: str) -> str:
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\n+', '\n', text)
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return '\n'.join(lines)

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
            service_logger.info(f"[START] QueryId={self.query_id}, Temperature={self.temperature},"
                                f"MaxTokens={self.max_new_tokens},RawText={self.url}")
            
            if not self.url:
                message = ErrorMessages.MISSING_PARMS
                error_message = f"{message} url  not provided."
                service_logger.warning(f"[INIT]{error_message}")
                return _response(HttpStatusCodes.BAD_REQUEST, message, error_message)
            try:
                loader = WebBaseLoader(self.url,verify_ssl=False,encoding='utf-8')
                docs = loader.load()
                content=docs[0].page_content
            except Exception as e:
                message = ErrorMessages.INTERNAL_SERVER_ERROR
                error_message= f"{message} {str(e)}"
                service_logger.error(error_message)
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,message,error_message)
            
            cleaned_text=self.simple_clean_text(content)

            message_list = [
                {"role": self.ROLE_SYSTEM, "content": f":{cleaned_text}"},
                {"role": self.ROLE_USER, "content": WEBSITE_PROMPT2}
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
                            print(result, end="", flush=True)
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