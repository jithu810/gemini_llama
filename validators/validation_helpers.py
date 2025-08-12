from utils.service_validators import validate_doc
from utils.messages import ErrorMessages,ShortMessages
from utils.status_codes import HttpStatusCodes
from utils.response_utils import log_and_respond

def validate_doc_input(pdf_path, query_id, logger):
    # Check for required parameters
    if not pdf_path or not query_id :
        return log_and_respond(logger, HttpStatusCodes.BAD_REQUEST, ErrorMessages.MISSING_PARMS,f"{ShortMessages.VALIDATION_ERROR} PDF path or QueryId not provided.")

    is_valid, file_type_or_error = validate_doc(pdf_path)
    if not is_valid:
        return log_and_respond(logger, HttpStatusCodes.BAD_REQUEST, ErrorMessages.ERROR_VAL_DOC, f"{ShortMessages.VALIDATION_ERROR}{file_type_or_error}")
    return None,file_type_or_error  # No errors
