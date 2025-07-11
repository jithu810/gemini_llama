from core.document_extractor import DocumentExtractor
from utils.utils import create_folder
from utils.config import Config
from models.model_manager import ModelManager

# Constants
OUTPUT_DIR = "content/extracted"
create_folder(OUTPUT_DIR)

loggers = Config.init_logging()
# Logger
logger = loggers['handlerfile']

class DocumentProcessor:
    """Handles document extraction and processing."""
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def process_document(self, page_number):
        """Extracts images from a scanned document."""
        logger.info(f"Processing document: {self.pdf_path} on page {page_number}")
        if not self.pdf_path:
            error_msg = "PDF path is not provided."
            logger.error(error_msg)
            return None, None, error_msg
        try:
            # Validate PDF file
            invoice_query_processor = DocumentExtractor(self.pdf_path, OUTPUT_DIR)
            image_paths, unique_folder_name, error = invoice_query_processor.process_invoice(page_number)
            if not image_paths and not unique_folder_name:
                error_msg = "No images extracted from the PDF or unique folder name not generated."
                logger.error(error_msg)
                return None, None, error_msg
            if error:
                logger.error(f"Error during invoice processing: {error}")
                return None, None, error  # Return None and the error message, plus a placeholder for unique_folder_name
            if not image_paths:
                error_msg = "No images extracted from the PDF."
                logger.error(error_msg)
                return None, None, error_msg  # Return None and the error message, plus a placeholder for unique_folder_name
            logger.info(f"Extracted images saved successfully: {image_paths}")
            return image_paths, unique_folder_name, None  # Return image paths, folder name, and no error
        except Exception as e:
            error_msg = f"Exception during document processing: {str(e)}"
            logger.exception(error_msg)
            return None, None, error_msg  # Return None and the exception message, plus a placeholder for unique_folder_name

class AIProcessor:
    """
    AIProcessor is responsible for handling AI model processing tasks related to text extraction from images.
    Attributes:
        model_key (str): The identifier for the AI model to be used (default: "minicpm").
        model: The loaded AI model instance.
        error: Error message encountered during model loading, if any.
    Methods:
        __init__(model_key="minicpm"):
            Initializes the AIProcessor by loading the specified AI model. Raises an exception if model loading fails.
        extract_text_from_images(prompt, image_paths, temperature=0.3, max_new_tokens=128, query_id=None, use_history=False, stream=False):
            Extracts text from the provided images using the loaded AI model and the given prompt.
            Parameters:
                prompt (str): The prompt or instruction for the AI model.
                image_paths (list): List of file paths to the images for text extraction.
                temperature (float): Sampling temperature for the model's output (default: 0.3).
                max_new_tokens (int): Maximum number of tokens to generate (default: 128).
                query_id (optional): Identifier for the query, if needed.
                use_history (bool): Whether to use conversation history (default: False).
                stream (bool): Whether to stream the output (default: False).
            Returns:
                dict: A dictionary containing the extraction results under "results" and any error message under "error".
    """
    def __init__(self, model_key="minicpm"):
        """Initializes the AIProcessor with the specified model."""
        self.model_key = model_key
        self.model = None
        self.model, self.error = ModelManager.get_model(self.model_key)
        if self.error:
            logger.error(f"[AIProcessor INIT] Model loading error: {self.error}")
            raise Exception(f"Model loading failed: {self.error}")
        logger.info(f"[AIProcessor INIT] Model {self.model_key} loaded successfully.")

    def extract_text_from_images(self,prompt,image_paths,temperature=0.3,max_new_tokens=128,query_id=None,use_history=False,stream=False):
        """Uses MiniCPM model to extract text from images."""
        logger.info(f"[AIProcessor EXTRACT] Extracting text from images with prompt: {prompt}, image_paths: {image_paths}, temperature: {temperature},"
                    f" max_new_tokens: {max_new_tokens}, query_id: {query_id}, use_history: {use_history},stream:{stream} ")
        if not self.model:
            logger.error("[AIProcessor EXTRACT] Model is not loaded.")
            return {"results": None, "error": "Model is not loaded."}
        try:
            extracted_text = self.model.extract_text_from_images(
                prompt, image_paths, OUTPUT_DIR, temperature=temperature, max_new_tokens=max_new_tokens,query_id=query_id,use_history=use_history,stream=stream)
            logger.info(f"[AIProcessor EXTRACT] Text extraction completed. Response: {extracted_text}")
            # Validate the response format
            if not extracted_text or not isinstance(extracted_text, dict):
                logger.error(f"[AIProcessor EXTRACT] Invalid response format: {extracted_text}")
                return {"results": None, "error": "Invalid response format from model."}
            return {"results": extracted_text, "error": None}
        except Exception as e:
            logger.exception(f"Exception during extract_text_from_images: {str(e)}")
            return {"results": None, "error": str(e)}
