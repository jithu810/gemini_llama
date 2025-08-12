# -*- coding: utf-8 -*-
from utils.messages import ErrorMessages, SuccessMessages,ShortMessages
from utils.status_codes import HttpStatusCodes
from utils.timer import Timer
from utils.prompts import REFINE_TEXT_PROMPT
from services.llama.base_processor import BaseDocumentProcessor
import re


class EmailRefineProcessor(BaseDocumentProcessor):
    def __init__(self, params: dict, context):
        super().__init__(params, context)
        self.log = self.logger['document_summary']
        self.log.info(f"{ShortMessages.INVOKED} DocumentSummaryProcessor")
        self.params = params
        self.context = context

        self.raw_text = self.params.get("RawText")  
        
    def strip_html_tags(self,text):
        clean_text = re.sub(r'<[^>]+>', '', text)
        return clean_text

    def process(self):
       
        try:
            self.log.info(f"{ShortMessages.PARAMS} QueryId={self.query_id}, Temperature={self.temperature},"
                                f"MaxTokens={self.max_new_tokens},RawText={self.raw_text}")
            # Determine user message
            self.raw_text=self.strip_html_tags(self.raw_text)
            message_list = [
                {"role": self.ROLE_SYSTEM, "content": f":{self.raw_text}"},
                {"role": self.ROLE_USER, "content": REFINE_TEXT_PROMPT}
            ]
            self.log.info("Calling the LlamaSummarizer for document summarization...")
            summary_accumulator = ""

            with Timer() as total_timer:
                for result in self.llama_model.stream_summary(max_new_tokens=self.max_new_tokens, temperature=self.temperature, messages=message_list,query_id=self.query_id,use_history=False):
                    if isinstance(result, dict) and "error" in result:
        
                        return self.respond(self.log, 
                                    HttpStatusCodes.INTERNAL_SERVER_ERROR,
                                    ErrorMessages.SUMMARY_FAILED, 
                                    f":{str(result['error'])}")
                    else:
                        if self.production !='production':
                            print(result, end="", flush=True)
                            # self.log.info(f"Streamed chunk: {result.strip()}")
                        summary_accumulator += result
            result = {
                "QueryId": self.query_id,
                "Time": f"{total_timer.interval:.2f}s",
                "Data": summary_accumulator
            }
            self.log.info(f"[DATA SENT]{result}")
            return {
                "status_code": HttpStatusCodes.OK,
                "status_description": "OK",
                "remarks": SuccessMessages.PROCESSED_SUCCESSFULLY,
                "data": result
            }

        except Exception as e:
                return self.respond(self.log, 
                                HttpStatusCodes.INTERNAL_SERVER_ERROR, 
                                ErrorMessages.INTERNAL_SERVER_ERROR,
                                f":{str(e)}")