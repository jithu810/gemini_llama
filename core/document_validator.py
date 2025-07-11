import os 
from utils.doc_str_check import PDFTextChecker
from utils.utils import get_file_type
from utils.config import Config
loggers = Config.init_logging()

handler_logger = loggers['handlerfile']

class DocumentValidator:
    """Handles validation of documents before processing."""

    @staticmethod
    def validate_file(pdf_path):
        """
        Validates the existence and format of a given file path.

        Args:
            pdf_path (str): The path to the document file to validate.

        Returns:
            tuple:
                - success (bool): True if the file exists and is of a supported type; False otherwise.
                - message (str): A message indicating the result of the validation (success, not found, unsupported type, or error).
                - file_type (str or None): The detected file type if successful; otherwise, None.

        Supported file formats: PDF, DOC, DOCX, HTML, TXT, XLSX.
        """
        try:
            if not os.path.exists(pdf_path):
                handler_logger.warning(f"File not found: {pdf_path}")
                return False, f"File not found at path: {pdf_path}", None

            file_type = get_file_type(pdf_path)
            supported_formats = ["pdf", "doc", "docx", "html", "txt", "xlsx"]
            if file_type is None or not any(fmt in file_type.lower() for fmt in supported_formats):
                handler_logger.warning(f"Unsupported file type detected: {file_type}")
                return False, f"Unsupported file type: {file_type}", None

            return True, "File validated successfully", file_type

        except Exception as e:
            handler_logger.exception(f"Exception during file validation: {str(e)}")
            return False, f"Validation error: {str(e)}", None

    @staticmethod
    def check_text_match(pdf_path, expected_text):
        """Checks if the expected text is found in the PDF and returns error if any."""
        try:
            checker = PDFTextChecker()
            found_terms = checker.check_text_in_pdf(pdf_path, [expected_text])

            result = found_terms.get(expected_text)

            if result is None:
                handler_logger.warning(f"No result found for query '{expected_text}'")
                return {"match": False, "error": "No result found for the query term."}

            if "error" in result:
                handler_logger.error(f"Error during PDF processing: {result['error']}")
                return {"match": False, "error": result["error"]}

            if result.get("found", False):
                handler_logger.info(f"Text '{expected_text}' found directly in the PDF.")
                return {"match": True}

            handler_logger.info(f"Text '{expected_text}' NOT found.")
            return {"match": False}

        except Exception as e:
            error_msg = f"Exception while checking text match: {str(e)}"
            handler_logger.exception(error_msg)
            return {"match": False, "error": error_msg}


