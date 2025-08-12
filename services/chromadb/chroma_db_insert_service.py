
from datetime import datetime

from utils.service_validators import validate_doc
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.response_utils import response as _response
from utils.timer import Timer

from core.document_extractor import extract_text_from_file
from vector_database.connect import get_chroma_client
from interceptors.request_id_interceptor import request_id_ctx

from utils.config import Config
loggers= Config.init_logging()

service_logger = loggers['chromadb']

class ChromaInsertProcessor:
    def __init__(self, params: dict, context):
        """
        Initializes the ChromaInsertProcessor with parameters and context.
        :param params: Dictionary containing parameters for processing.
        :param context: Context for the service call.
        """
        service_logger.info("[INIT] ChromaInsertProcessor initialized with parameters.")
        self.params = params
        self.context = context

        self.query_id = self.params.get("QueryId")
        self.doc_path = self.params.get("FilePath")
        self.doc_source = self.params.get("Source") or self.params.get("source", "unknown")
        self.tags = self.params.get("Tags") or self.params.get("tags", [])
        self.uploaded_by = self.params.get("UploadedBy") or self.params.get("uploaded_by", "admin")  

        if not self.query_id:   
            service_logger.warning(f"[INIT] Missing QueryId parameter.")
            raise ValueError("Missing QueryId parameter.")
        request_id_ctx.set(self.query_id)

        try:
            self.client, self.embedding_func = get_chroma_client()
            service_logger.info("[CHROMA] Client and embedding function initialized successfully.")
        except RuntimeError as err:
            service_logger.error(f"[CHROMA ERROR] {err}")
            self.client = None
            self.embedding_func = None

    def chunk_text(self,text, max_chunk_size=1000):
        """
        Splits text into chunks of max_chunk_size (characters).
        """
        service_logger.info(f"[CHUNK TEXT] Chunking text of length {len(text)} with max chunk size {max_chunk_size}")
        words = text.split()
        chunks, current_chunk = [], []
        for word in words:
            # Calculate length if we add this word (+1 for space)
            if sum(len(w) + 1 for w in current_chunk) + len(word) + 1 <= max_chunk_size:
                current_chunk.append(word)
            else:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                service_logger.debug(f"[CHUNK TEXT] Created chunk of length {len(chunk_text)}")
                current_chunk = [word]

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            service_logger.debug(f"[CHUNK TEXT] Created final chunk of length {len(chunk_text)}")
        service_logger.info(f"[CHUNK TEXT] Total chunks created: {len(chunks)}")
        return chunks

    def process(self):
        try:
            service_logger.info(
                f"[START] Uploading document to Chroma. FilePath={self.doc_path}, Id={self.query_id}, "
                f"Source={self.doc_source}, Tags={self.tags}, UploadedBy={self.uploaded_by}"
            )
            if not self.doc_path:
                messages= ErrorMessages.MISSING_PARMS
                error_message = f"{messages} PDF path not provided."
                service_logger.warning(f"[MISSING] {error_message}")
                return _response(HttpStatusCodes.BAD_REQUEST,messages,error_message)
            if not self.query_id:
                messages = ErrorMessages.MISSING_PARMS
                error_message = f"{messages} QueryId not provided."
                service_logger.warning(f"[MISSING] {error_message}")
                return _response(HttpStatusCodes.BAD_REQUEST, messages, error_message)
            if not self.doc_source:
                self.doc_source = "unknown"
                service_logger.warning("[MISSING] Source not provided, defaulting to 'unknown'.")   
            if self.client is None:
                messages = ErrorMessages.CLIENT_NOT_INITIALIZED
                error_message = f"{messages} ChromaDB client not initialized."
                service_logger.error(error_message)
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,messages,error_message)
            if self.embedding_func is None:
                messages = ErrorMessages.EMBEDDING_FUNC_NOT_INITIALIZED
                error_message = f"{messages} Embedding function not initialized."
                service_logger.error(error_message)
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, messages, error_message)
            is_valid, file_type_or_error = validate_doc(self.doc_path)
            if not is_valid:
                messages= ErrorMessages.ERROR_VAL_DOC
                error_message= f"{messages}: {file_type_or_error}"
                service_logger.warning(error_message)
                return _response(HttpStatusCodes.BAD_REQUEST,messages,error_message)
            service_logger.info(f"[VALIDATION SUCCESS] Document is valid: {file_type_or_error}")
            try:
                input_text = extract_text_from_file(self.doc_path, file_type_or_error)
                if not input_text or input_text.strip() == "":
                    raise ValueError("Extracted text is empty.")
            except Exception as e:
                messages = ErrorMessages.AI_EXTRACTION_FAILED
                error_message = f"{messages}: {str(e)}"
                service_logger.error(f"[EXTRACTION FAILED] {messages}: {e}")
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,messages,error_message)
            
            service_logger.info(f"[EXTRACTION SUCCESS] Extracted text of length {len(input_text)}")
            
            chunks = self.chunk_text(input_text)
            if not chunks:
                messages = ErrorMessages.EMPTY_DOCUMENT
                error_message = f"{messages}: No text extracted from the document."
                service_logger.warning(error_message)
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, messages, error_message)
            
            service_logger.info(f"[CHUNKING SUCCESS] Created {len(chunks)} chunks from the document.")

            # Create metadata
            metadata = {
                "ids": self.query_id,
                "source": self.doc_source,
                "type": file_type_or_error,
                "tags": ", ".join(self.tags) if isinstance(self.tags, list) else str(self.tags),
                "uploaded_by": self.uploaded_by,
                "uploaded_at": str(datetime.utcnow())
            }
            # Add to ChromaDB
            try:
                with Timer() as inser_timer:
                    collection = self.client.get_or_create_collection(name="uploaded_documents", embedding_function=self.embedding_func)
                    for idx, chunk in enumerate(chunks):
                        chunk_id = f"{self.query_id}_chunk_{idx}" if self.query_id else f"{self.doc_source}_{datetime.utcnow().timestamp()}_{idx}"
                        chunk_metadata = {
                            **metadata,
                            "chunk_index": idx
                        }
                        collection.add(
                            documents=[chunk],
                            metadatas=[chunk_metadata],
                            ids=[chunk_id]
                        )
                    service_logger.info(f"[CHROMA UPLOAD SUCCESS] Document stored with metadata: {metadata}")
                service_logger.info(f"[CHROMA UPLOAD SUCCESS] ChromaDB Insert Processor completed. Total time: {inser_timer.interval:.2f} seconds")
                return {
                    "status_code": HttpStatusCodes.OK,
                    "status_description": "OK",
                    "remarks": SuccessMessages.PROCESSED_SUCCESSFULLY,
                    "data": {
                        "ids": metadata["ids"],
                        "source": metadata["source"],
                        "uploaded_at": metadata["uploaded_at"],
                        "tags": metadata["tags"],
                        "chunks": len(chunks),
                        "Time": f"{inser_timer.interval:.2f}s",
                    }
                }

            except Exception as e:
                messages = ErrorMessages.CHROMA_UPLOAD_FAILED
                error_message = f"{messages}: {str(e)}"
                service_logger.error(error_message)
                return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,messages,error_message)

        except Exception as e:
            messages = ErrorMessages.INTERNAL_SERVER_ERROR
            error_message = f"{messages}: {str(e)}"
            service_logger.error(f"[INTERNAL SERVER ERROR] {error_message}")
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, messages, error_message)
