class ErrorMessages:
    """
    A collection of standardized error message constants for use throughout the application.
    """

    BAD_REQUEST="BAD REQUEST"
    MISSING_PARMS="Request missing params"
    ERROR_VAL_DOC="error validating document"
    PAGE_NOT_FOUND="page number not found in doc"
    IMAGE_EXTRACTION_FAILED = "Error extracting images from document"
    INTERNAL_SERVER_ERROR = "Internal server error"
    SCAN_CHECK_FAILED = "Error determining if document is scanned."
    IMAGE_TEXT_EXTRACTION_FAILED = "Text extraction from images failed."
    MODEL_FAILED_IMAGE = "Model failed to process image"
    AMOUNT_COMPARISON_FAILED="failed comparing amount"
    MODEL_LOAD_FAILED="Model load failed"
    SUMMARY_FAILED="error while summarizing"
    NO_IMAGES_FOUND= "No images found in the document"
    HISTORY_LOAD_FAILED= "Failed to load conversation history"
    HISTORY_SAVE_FAILED = "Failed to save conversation history"
    CONTEXT_FETCH_FAILED= "Failed to fetch context for the query"
    SYSTEM_PROMPT_NOT_FOUND= "System prompt not found for the query"
    CLIENT_NOT_INITIALIZED = "AI client not initialized"
    EMBEDDING_FUNC_NOT_INITIALIZED = "Embedding function not initialized"
    EMPTY_RESPONSE= "Empty response from AI model"
    CHROMA_UPLOAD_FAILED = "Failed to upload document to ChromaDB"
    EMPTY_DOCUMENT= "Document is empty or contains no text"
    FILE_TYPE_MISSMATCH="the uploaded file is not an excel type"

    INVALID_FILE = "Invalid document file"
    PAGE_COUNT_ERROR="Unable to retrieve PDF page count"
    AI_EXTRACTION_FAILED = "Error during  text extraction"
    EMPTY_FIELDS = "File path, prompt, or expected result is missing"
    SCAN_STATUS_UNKNOWN = "Unable to determine if the document is scanned"
    SCAN_STATUS_EXCEPTION = "Exception occurred while checking scan status"
    OUT_DIR_DEL_FAILED ="Error while deleting output directory"
    TIME_CAL_FAILED="Request processed in unknown time (Timer failed)"
    UNSUPPORTED_SERVICE = "Unsupported service_type"
    NUMBER_NORMALIZATION_FAILED = "Error normalizing number"

class SuccessMessages:
    """
    A collection of success message templates used throughout the application.

    Attributes:
        PROCESSED_SUCCESSFULLY (str): Message indicating successful processing.
        VALIDATION_SUCCESS (str): Message indicating validation has completed successfully.
        TIME_CAL_SUCCESS (str): Message indicating the request was processed, including the duration in seconds.
        DIR_CLEARED (str): Message indicating the output directory has been cleared.
        DIR_CREATED (str): Message indicating a directory was created, with the directory path.
        NUMBER_NORMALIZED (str): Message indicating a number was normalized, showing the original and normalized values.
        AMOUNT_EXTRACTED (str): Message indicating a value was extracted.
        AMOUNT_COMPARISON (str): Message comparing extracted and matched values.
        DIR_CLEARED_AND_REMOVED (str): Message indicating the processed folder was deleted.
    """
    PROCESSED_SUCCESSFULLY="Successfully processed"
    VALIDATION_SUCCESS = "Validation complete."
    TIME_CAL_SUCCESS = "Request processed in {duration:.4f} seconds"
    DIR_CLEARED = "Cleared output directory"
    DIR_CREATED = "The directory '{path}' was created."
    NUMBER_NORMALIZED = "Normalized '{original}' to {normalized}"
    AMOUNT_EXTRACTED = "Extracted value: {value}"
    AMOUNT_COMPARISON = "Extracted: {extracted}, Match: {match}"
    DIR_CLEARED_AND_REMOVED="Deleted the processed folder"