# tests/test_document_passport_service.py

import pytest
from unittest.mock import patch, MagicMock
from services.minicpm.document_passport_service import DocumentPassportProcessor
from utils.status_codes import HttpStatusCodes
from utils.messages import ErrorMessages,SuccessMessages

# Test process returns error if missing parameters
def test_process_missing_parameters(monkeypatch):
    params = {
        "FilePath": None,
        "Query": None,
        "PageNumber": 0,
        "QueryId": "id"
    }
    processor = DocumentPassportProcessor(params, context={})
    response = processor.process()
    assert response["status_code"] == "400"
    assert ErrorMessages.MISSING_PARMS in response["status_description"]

# Helper to create default params dict
def default_params():
    return {
        "FilePath": "dummy.pdf",
        "Query": ["What is the document?"],
        "PageNumber": 0,
        "QueryId": "1234",
        "temperature": 0.7,
        "max_new_tokens": 100
    }

# # Test process returns error if PDF validation fails
@patch("DocumentPassportProcessor.validate_pdf_file", return_value=(False, "Not a PDF"))
def test_process_pdf_validation_fail(mock_validate_pdf):
    processor = DocumentPassportProcessor(default_params(), context={})
    response = processor.process()
    assert response["status_code"] == 400
    assert ErrorMessages.ERROR_VAL_DOC in response["status_description"]