# -*- coding: utf-8 -*-
import os
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.timer import Timer
from utils.response_utils import response as _response
from interceptors.request_id_interceptor import request_id_ctx
from utils.value_check import safe_int,safe_float
from utils.config import Config
from utils.service_validators import validate_pdf_file
from sentence_transformers import SentenceTransformer,util
import pandas as pd
import os
import json

loggers = Config.init_logging()

OUTPUT_DIR= Config.OUTPUT_DIR
TEMPERATURE=Config.TEMPERATURE
MAX_NEW_TOKENS=Config.MAX_NEW_TOKENS
PRODUCTION=Config.ENVIRONMENT

service_logger = loggers['document_summary']

class ExcelColumnCompareProcessor:
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
        self.raw_text = self.params.get("RawText") 
        self.doc_path = self.params.get("FilePath")
        self.temperature=safe_float(self.params.get("temperature",TEMPERATURE),TEMPERATURE)
        self.max_new_tokens=safe_int(self.params.get("max_new_tokens",MAX_NEW_TOKENS),MAX_NEW_TOKENS)

        if not self.query_id:   
            service_logger.warning(f"[INIT] Missing QueryId parameter.")
        request_id_ctx.set(self.query_id)
        
    def read_excel_with_detected_header(self,filepath, preview_rows=10):
        """
        Reads an Excel file and tries to automatically detect the header row
        based on the first row that contains only strings.
        Args:
            filepath (str): Path to the Excel file.
            preview_rows (int): Number of initial rows to scan for the header.
        Returns:
            pd.DataFrame: DataFrame loaded using the detected header row.
        """
        try:
            # Step 1: Preview without headers
            preview = pd.read_excel(filepath,header=None,nrows=preview_rows)
            # Step 2: Heuristic to detect the header row
            header_row = None
            for i, row in preview.iterrows():
                if all(isinstance(x, str) for x in row):
                    header_row = i
                    break
            if header_row is None:
                raise ValueError("No suitable header row detected.")
            # Step 3: Read again with detected header row
            df = pd.read_excel(filepath,header=header_row)
            return df,header_row
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return None

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
                                f"MaxTokens={self.max_new_tokens},RawText={self.raw_text}")
            # Determine user message
            
            if not self.doc_path:
                message = ErrorMessages.MISSING_PARMS
                error_message = f"{message} PDF path not provided."
                service_logger.warning(f"[INIT]{error_message}")
                return _response(HttpStatusCodes.BAD_REQUEST, message, error_message)
            
            is_valid,file_type_or_error = validate_pdf_file(self.doc_path)
            if not is_valid:
                message= ErrorMessages.ERROR_VAL_DOC
                error_message = f"[NOT VALID]{message} {file_type_or_error}"
                service_logger.warning(error_message)
                return _response(HttpStatusCodes.BAD_REQUEST,message,error_message)
            
            if file_type_or_error not in ["xlsx","xls"]:
                message = ErrorMessages.FILE_TYPE_MISSMATCH
                error_message = f"{message} - check the file."
                service_logger.warning(f"[INVALID FILE]{error_message}")
                return _response(HttpStatusCodes.BAD_REQUEST, message, error_message)

            df, header_row = self.read_excel_with_detected_header(self.doc_path)
            if df is None:
                message = ErrorMessages.ERROR_VAL_DOC
                error_message = "Failed to read the Excel file with a detected header."
                service_logger.error(f"{message}{error_message}")
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, message, error_message)
            
            # Get Excel column list
            excel_columns = df.columns.tolist()

            # Validate RawText
            if not isinstance(self.raw_text, list):
                message = ErrorMessages.MISSING_PARMS
                error_message = "RawText should be a list of column names."
                service_logger.warning(error_message)
                return _response(HttpStatusCodes.BAD_REQUEST,message,error_message)
   
            list1 = [f"This column represents {col.lower()}" for col in self.raw_text]
            list2 = [f"This column represents {col.lower()}" for col in excel_columns]
 
            with Timer() as total_timer:
                model = SentenceTransformer(r'Weights\all-mpnet-base-v2')
                # Encode and compute cosine similarity
                embeddings1 = model.encode(list1,convert_to_tensor=True)
                embeddings2 = model.encode(list2,convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(embeddings1,embeddings2)  

                used = set()
                matched_columns = []
                unmatched_columns = []
                threshold = safe_float(self.params.get("similarity_threshold",0.75),0.75)

                for i, original_col1 in enumerate(self.raw_text):
                    best_score = -1
                    best_idx = -1
                    for j,original_col2 in enumerate(excel_columns):
                        if j in used:
                            continue
                        score = cosine_scores[i][j].item()
                        if score > best_score:
                            best_score = score
                            best_idx = j
                    if best_score >= threshold:
                        used.add(best_idx)
                        matched_columns.append({
                            "source_column": original_col1,
                            "target_column": excel_columns[best_idx],
                            "similarity_score": round(best_score,4)
                        })
                    else:
                        unmatched_columns.append({
                            "source_column": original_col1,
                            "best_match_column": excel_columns[best_idx],
                            "max_similarity_score": round(best_score,4)
                        })
                output = {
                    "matched_columns": matched_columns
                    # "unmatched_columns": unmatched_columns
                }
            output=json.dumps(output)
            result = {
                "QueryId": self.query_id,
                "Time": f"{total_timer.interval:.2f}s",
                "header_row":header_row+1,
                "columns":excel_columns,
                "Data": output
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

