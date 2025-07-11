from core.document_handler import DocumentProcessor
from core.document_validator import DocumentValidator
from utils.config import Config
from utils.messages import ErrorMessages 

loggers= Config.init_logging()
utils_logger = loggers['utils']

def validate_pdf_file(pdf_path):
    """
    Validates a document file using DocumentValidator.

    Args:
        pdf_path (str): The path to the document file to validate.

    Returns:
        tuple:
            - success (bool): True if the file is valid and supported; False otherwise.
            - result (str): 
                - If valid: the detected file type (e.g., 'pdf', 'docx').
                - If invalid: an error message describing the failure reason 
                  (e.g., file not found, unsupported type, internal error).

    Notes:
        - This function wraps the DocumentValidator.validate_file method.
        - It logs errors using the utils_logger and handles exceptions gracefully.
    """
    try:
        valid, error_msg, file_type = DocumentValidator.validate_file(pdf_path)
        if not valid:
            utils_logger.error(f"{ErrorMessages.INVALID_FILE}: {error_msg}")
            return False, error_msg  # Return the actual error message
        return True, file_type
    except Exception as e:
        utils_logger.exception(f"{ErrorMessages.INTERNAL_SERVER_ERROR}: {str(e)}")
        return False, f"Internal server error: {str(e)}"

def extract_images_from_pdf(pdf_path, page_number):
    """
    Extracts images from a specified page of a PDF file.
    Args:
        pdf_path (str): The file path to the PDF document.
        page_number (int): The page number from which to extract images.
    Returns:
        tuple:
            - image_paths (list or None): List of file paths to the extracted images, or None if extraction failed.
            - unique_folder_name (str or None): Name of the folder where images are stored, or None if extraction failed.
            - error_msg (str or None): Error message if extraction failed, otherwise None.
    Logs:
        - Logs errors and exceptions encountered during image extraction.
    Raises:
        - Does not raise exceptions; all errors are logged and returned as part of the result tuple.
    """
    try:
        # Create DocumentProcessor instance
        processor = DocumentProcessor(pdf_path)

        # Process document and extract images
        image_paths, unique_folder_name, error_msg = processor.process_document(page_number)

        # If an error occurred during document processing, log it and return None
        if error_msg:
            utils_logger.error(f"Image extraction failed for {pdf_path}: {error_msg}")
            return None, None, error_msg

        # If no images were extracted, log the error and return None
        if not image_paths:
            error_msg = "No images extracted from the PDF."
            utils_logger.error(f"{ErrorMessages.IMAGE_EXTRACTION_FAILED}: {pdf_path} - {error_msg}")
            return None, None, error_msg

        # Return the image paths and unique folder name if successful
        return image_paths, unique_folder_name, None

    except Exception as e:
        # Log any exception that occurs during the process
        utils_logger.exception(f"{ErrorMessages.INTERNAL_SERVER_ERROR}: {str(e)}")
        return None, None, f"Exception occurred during image extraction: {str(e)}"