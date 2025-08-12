import magic
import logging
import pdfplumber
import shutil
import traceback
import os 
import re
from utils.config import Config
from typing import Optional, Union,Tuple
from utils.messages import ErrorMessages, SuccessMessages
from PyPDF2 import PdfReader

loggers = Config.init_logging()
utils_logger = loggers['utils']

def get_pdf_page_count(file_path):
    try:
        print(file_path)
        reader = PdfReader(file_path)
        return len(reader.pages), None
    except Exception as e:
        error_msg = f"{ErrorMessages.INTERNAL_SERVER_ERROR}: {str(e)}"
        utils_logger.exception(error_msg)
        return None, error_msg

def clear_output_directory(output_dir):
    """
    Removes all files and subdirectories inside the specified output directory,
    then deletes the output directory itself.
    Args:
        output_dir (str): Directory to clear and delete.
    """
    if os.path.exists(output_dir):
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                utils_logger.error(f"{ErrorMessages.OUT_DIR_DEL_FAILED}: {file_path} - {e}")
        try:
            shutil.rmtree(output_dir)
            utils_logger.info(f"{SuccessMessages.DIR_CLEARED_AND_REMOVED}:{output_dir}")
        except Exception as e:
            utils_logger.error(f"{SuccessMessages.DIR_CLEARED_AND_REMOVED}{output_dir} - {e}")
    else:
        utils_logger.warning(f"Output directory does not exist: {output_dir}")

def get_file_type(file_path):
    """
    Determines the simplified file type of a given file using its MIME type.

    Args:
        file_path (str): The path to the file whose type needs to be detected.

    Returns:
        str or None: 
            - A simplified file type string such as "pdf", "docx", "doc", "html", "txt", or "xlsx" 
              if the file matches a known format.
            - The raw MIME type string if it's unrecognized but detectable.
            - None if an error occurs during detection.

    Notes:
        - Uses the `python-magic` library to inspect the file's MIME type.
        - Logs and returns None in case of failure or exception.
    """
    try:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(file_path)

        if "pdf" in mime_type:
            return "pdf"
        elif "officedocument.wordprocessingml.document" in mime_type:
            return "docx"
        elif "msword" in mime_type:  # legacy .doc files
            return "doc"
        elif "html" in mime_type:
            return "html"
        elif mime_type.startswith("text/"):
            return "txt"
        elif "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in mime_type:
            return "xlsx"
        else:
            return mime_type  # fallback
        
    except Exception as e:
        utils_logger.error(f"Error checking filetype get_file_type: {e}")
        return None
    
def is_scanned_doc(file_path):
    """Check if the PDF is scanned by detecting text."""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    return False  
        return True 
    except Exception as e:
        utils_logger.error(f"Error checking scanned PDF is_scanned_doc: {e}")
        return None 

def get_extracted_texts(output_dir):
    """
    Reads extracted text from text files in output_dir and returns a list of (file_name, text).
    Args:
        output_dir (str): Directory containing extracted text files.
    Returns:
        list of tuples: [(file_name, extracted_text), ...]
    """
    extracted_data = []
    for file_name in sorted(os.listdir(output_dir)): 
        if file_name.endswith(".txt"):  
            file_path = os.path.join(output_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read().strip()
                    extracted_data.append((file_name, text))
            except Exception as e:
                utils_logger.error(f"Error reading file {file_name}: {e}")
    return extracted_data

def create_folder(path):
    try:
        if os.path.exists(path):
            return False
        else:
            os.makedirs(path, exist_ok=True)
            utils_logger.info(f"The directory '{path}' was created.")
            return True
    except Exception as e:
        utils_logger.error(f"Error creating folders: {e}")
        return None
    
def normalize_number(number_str: str) -> Tuple[Optional[Union[float, int]], Optional[str]]:
    """
    Convert a number string with commas into a valid float-compatible format.
    Returns a tuple: (normalized_number, error_message).
    """
    try:
        if not isinstance(number_str, str):
            msg = f"Expected string input but got {type(number_str)}: {number_str}"
            utils_logger.warning(msg)
            return None,msg

        number_str = number_str.strip().replace(",", "")
        if not number_str:
            msg = "Received an empty or whitespace-only string."
            utils_logger.warning(msg)
            return None,msg

        num = float(number_str)
        normalized_num = int(num) if num.is_integer() else num
        utils_logger.info(f"Normalized '{number_str}' to {normalized_num}")
        return normalized_num,None

    except Exception as e:
        error_details = traceback.format_exc()
        error_msg = f"{ErrorMessages.NUMBER_NORMALIZATION_FAILED}: {e}\n{error_details}"
        utils_logger.error(error_msg)
        return None, error_msg
    
def extract_amount(amount_str: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract the numeric value from the string (removes non-numeric characters and commas).
    Args:
        amount_str (str): The string containing the amount (e.g., 'AED 24,150.00').
    Returns:
        tuple: (value, error)
            - value: The extracted numeric value, or None if extraction fails.
            - error: An error message if something went wrong, or None if successful.
    """
    try:
        utils_logger.info(f"Extracting amount from the predicted text: '{amount_str}'")
        if not isinstance(amount_str, str):
            warning_msg = f"Skipping invalid input: {amount_str} (not a string)"
            utils_logger.warning(warning_msg)
            return None,warning_msg

        match = re.search(r"[\d,]+(?:\.\d+)?", amount_str)  # Match a number with optional commas and decimals
        if match:
            extracted_value, norm_error = normalize_number(match.group())
            if norm_error:
                return None,f"Normalization failed for extracted value: {norm_error}"
            utils_logger.info(f"Extracted value: {extracted_value}")
            return extracted_value,None

        warning_msg = f"Failed to extract numeric value from: '{amount_str}'"
        utils_logger.warning(warning_msg)
        return None,warning_msg

    except Exception as e:
        error_details = traceback.format_exc()
        error_msg = f"Error in extract_amount: {e}\n{error_details}"
        utils_logger.error(error_msg)
        return None,error_msg

def compare_amount(amount_str: str, target_value: Union[str, float]) -> Tuple[Optional[float], bool, Optional[str]]:
    """
    Compare the extracted amount from the string with a target float value.
    Args:
        amount_str (str): The string containing the amount (e.g., 'AED 24,150.00').
        target_value (str or float): The target value as a string or float.
    Returns:
        tuple: (extracted_value, is_match, error_message)
            - extracted_value (float or None)
            - is_match (bool): True if extracted amount matches target value
            - error_message (str or None): If any internal error occurred
    """
    try:
        utils_logger.info(f"Comparing predicted amount: '{amount_str}' with target value given: {target_value}")
        target_value_norm, target_error = normalize_number(str(target_value))
        if target_error:
            utils_logger.error(f"Normalization failed for target value: {target_error}")
            return None,False,target_error

        extracted_amount, extract_error = extract_amount(amount_str)
        if extract_error:
            warning_msg = f"Amount extraction failed: {extract_error}"
            utils_logger.warning(warning_msg)
            return None,False,warning_msg

        is_match = abs(extracted_amount - target_value_norm) < 0.01  # tolerance for float comparison
        utils_logger.info(f"Extracted: {extracted_amount}, Match: {is_match}")
        return extracted_amount,is_match,None

    except Exception as e:
        error_details = traceback.format_exc()
        error_msg = f"Error in compare_amount: {e}\n{error_details}"
        utils_logger.error(error_msg)
        return None,False,error_msg
    
def cleanup_folder(folder_name,OUTPUT_DIR):
        """
        Deletes a temporary folder within the specified output directory.

        Args:
            folder_name (str): The name of the folder to be deleted.
            OUTPUT_DIR (str): The path to the output directory containing the folder.

        Logs:
            Info: Logs successful deletion of the temporary folder.
            Error: Logs an error message if deletion fails.
        """
        try:
            delete_path = os.path.join(OUTPUT_DIR, folder_name)
            clear_output_directory(delete_path)
            utils_logger.info(f"[CLEANUP] Deleted temporary folder: {delete_path}")
        except Exception as e:
            utils_logger.error(f"[CLEANUP ERROR] {ErrorMessages.OUT_DIR_DEL_FAILED}: {e}")