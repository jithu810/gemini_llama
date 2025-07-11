import fitz
import logging 
import re
from utils.config import Config
loggers = Config.init_logging()

utils_logger = loggers['utils']

class PDFTextChecker:
    def __init__(self):
        pass
    
    @staticmethod
    def normalize_number(text):
        """Remove commas and standardize numbers to float format where possible."""
        text = text.replace(",", "")  # Remove commas
        return re.sub(r'(\d+\.\d*?)0+$', r'\1', text)  # Remove trailing zeros after decimal
    
    def check_text_in_pdf(self, pdf_path, search_terms):
        """
        Checks for exact matches of search terms in a PDF.
        Returns:
            dict: {
                term: {
                    "found": bool,
                    "count": int,
                    (optional) "error": str
                }
            }
        """
        found_terms = {term: {'found': False, 'count': 0} for term in search_terms}

        try:
            utils_logger.info(f"Starting PDF text check: {pdf_path}")
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc, start=1):
                try:
                    page_text = page.get_text("text")
                    normalized_text = self.normalize_number(page_text.lower())
                    for term in search_terms:
                        normalized_term = self.normalize_number(term.lower())
                        count = normalized_text.count(normalized_term)

                        if count > 0:
                            found_terms[term]['count'] += count
                            utils_logger.info(f"[PAGE {page_num}] Found '{term}' {count} time(s)")

                except Exception as page_error:
                    utils_logger.warning(f"Error reading page {page_num}: {page_error}")
                    continue

            # Post-processing: update "found" status
            for term in search_terms:
                count = found_terms[term]['count']
                if count > 3:
                    found_terms[term]['found'] = False
                    utils_logger.warning(f"'{term}' found more than 3 times ({count}) → Marked as NOT found.")
                elif count > 0:
                    found_terms[term]['found'] = True
            return found_terms

        except Exception as e:
            utils_logger.error(f"Exception reading PDF {pdf_path}: {e}", exc_info=True)
            return {
                term: {
                    'found': False,
                    'count': 0,
                    'error': f"PDF error: {str(e)}"
                }
                for term in search_terms
            }
