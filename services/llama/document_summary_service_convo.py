# -*- coding: utf-8 -*-
import os
from utils.messages import ErrorMessages, SuccessMessages,ShortMessages
from utils.status_codes import HttpStatusCodes
from utils.timer import Timer
from utils.prompts import SUMMARIZE_PROMPT,DEFAULT_SUMMARIZE_PROMPT
from core.document_extractor import extract_text_from_file
from services.llama.base_processor import BaseDocumentProcessor

class DocumentSummaryConvoProcessor(BaseDocumentProcessor):
    def __init__(self, params: dict, context):
        super().__init__(params, context)

        self.log = self.logger['document_summary']
        self.log.info(f"{ShortMessages.INVOKED} DocumentSummaryConvoProcessor")
        self.queries = self.params.get("Query")
        self.raw_text = self.params.get("RawText")
        
    def process(self):
        
        try:
            self.log.info(f"{ShortMessages.PARAMS} QueryId={self.query_id}, FilePath={self.doc_path}, Temperature={self.temperature},"
                                f"MaxTokens={self.max_new_tokens},RawText={self.raw_text}")
            
            if self.raw_text and self.raw_text.strip():
                input_text = self.raw_text.strip()
                self.log.info("Using provided raw_text for summarization.")
            else:    

                # Common validations
                validation_response,file_type_or_error = self.validate_input(self.doc_path, self.query_id, self.log)
                if validation_response:
                    return validation_response
                
                try:
                    input_text = extract_text_from_file(self.doc_path,file_type_or_error)
                    if not input_text or input_text.strip() == "":
                        raise ValueError("Extracted text is empty.")
                except Exception as e:
                    return self.respond(self.log,
                                            HttpStatusCodes.BAD_REQUEST,
                                            ErrorMessages.AI_EXTRACTION_FAILED,
                                            f"{e} text extraction failed")
            # Determine user message
            user_message = self.queries[0] if isinstance(self.queries, list) and self.queries else DEFAULT_SUMMARIZE_PROMPT
            message_list = [
                {"role": self.ROLE_SYSTEM, "content": f"{SUMMARIZE_PROMPT}:\n\n{input_text}"},
                {"role": self.ROLE_USER, "content": user_message}
            ]

            self.log.info("Calling the LlamaSummarizer for document summarization...")
            summary_accumulator = ""

            with Timer() as total_timer:
                for result in self.llama_model.stream_summary(max_new_tokens=self.max_new_tokens, temperature=self.temperature, messages=message_list,query_id=self.query_id,use_history=True):
                    if isinstance(result, dict) and "error" in result:
                        return self.respond(self.log, 
                                    ErrorMessages.SUMMARY_FAILED, 
                                    f":{str(result['error'])}")
                    else:
                        if self.production !='production':
                            print(result,end="",flush=True)
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