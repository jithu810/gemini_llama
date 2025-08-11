from utils.utils import get_pdf_page_count
from utils.service_validators import validate_pdf_file
from utils.messages import ErrorMessages,ShortMessages
from utils.status_codes import HttpStatusCodes
from utils.response_utils import log_and_respond

def validate_pdf_input(pdf_path, query_id, logger):
    # Check for required parameters
    if not pdf_path or not query_id :
        return log_and_respond(logger, HttpStatusCodes.BAD_REQUEST, ErrorMessages.MISSING_PARMS,f"{ShortMessages.VALIDATION_ERROR} PDF path or QueryId not provided.")

    is_valid, file_type_or_error = validate_pdf_file(pdf_path)
    if not is_valid:
        return log_and_respond(logger, HttpStatusCodes.BAD_REQUEST, ErrorMessages.ERROR_VAL_DOC, f"{ShortMessages.VALIDATION_ERROR}{file_type_or_error}")
    
    total_pages, error_msg = get_pdf_page_count(pdf_path)
    if total_pages is None:
        return log_and_respond(logger, HttpStatusCodes.INTERNAL_SERVER_ERROR, ErrorMessages.PAGE_COUNT_ERROR,f"{ShortMessages.VALIDATION_ERROR} {error_msg}")
    
    return None,file_type_or_error  # No errors
