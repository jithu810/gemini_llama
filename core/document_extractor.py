import traceback
from core.document_processor import DocumentProcessor
from utils.config import Config
from bs4 import BeautifulSoup
import docx2txt
from PyPDF2 import PdfReader
import re
import pandas as pd   
from nltk.tokenize import sent_tokenize, word_tokenize
from io import StringIO
loggers= Config.init_logging()

document_extractor_logger = loggers['document_extractor']

class DocumentExtractor:
    def __init__(self, pdf_path, output_dir):
        """
        Initialize the InvoiceQueryProcessor with the path to the PDF and the output directory.
        Args:
            pdf_path (str): Path to the PDF file.
            output_dir (str): Directory to save the extracted text.
        """
        document_extractor_logger.info(f"Initializing DocumentExtractor with PDF path: {pdf_path} and output directory: {output_dir}")
        if not pdf_path:
            error_msg = "PDF path is not provided."
            document_extractor_logger.error(error_msg)
            raise ValueError(error_msg)
        if not output_dir:
            error_msg = "Output directory is not provided."
            document_extractor_logger.error(error_msg)
            raise ValueError(error_msg)
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        # Initialize the DocumentProcessor with the provided PDF path and output directory
        self.document_processor = DocumentProcessor(pdf_path, output_dir)
        document_extractor_logger.info(f"DocumentExtractor initialized successfully with path '{pdf_path}' and output folder '{output_dir}' ")

    def read_text_file(self, file_path):
        """
        Read the content of a text file.
        Args:
            file_path (str): The path to the text file to read.
        Returns:
            str: The content of the text file.
        """
        if not file_path:
            error_msg = "File path is not provided."
            document_extractor_logger.error(error_msg)
            raise ValueError(error_msg)
        document_extractor_logger.info(f"Reading text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            document_extractor_logger.debug(f"Successfully opened file: {file_path}")
            return file.read()

    def process_queries(self,queries,filenames):
        """
        Process a list of queries on the extracted text from the specified files.
        Args:
            filenames (list of str): The list of filenames to process.
            queries (list of str): The list of queries to run on the extracted text.
        """
        for filename in filenames:
            text = self.read_text_file(filename)
            for query in queries:
                result = self.text_processor.process_text_with_model(text, query)
                document_extractor_logger.info(f"Result for file '{filename}' with query '{query}': {result}")

    def process_invoice(self, page_number=None):
        """
        Process the invoice PDF to extract text and run queries.
        Args:
            page_number (int, optional): Specific page number to extract text from.
        Returns:
            tuple: (image_paths, unique_folder_name, error_message), where error_message is None if no error occurred.
        """
        document_extractor_logger.info(f"Processing invoice PDF: {self.pdf_path} on page {page_number}")
        try:
            # Step 1: Load pages and check for retrieval errors
            if page_number is not None:
                document_extractor_logger.info(f"Processing specific page: {page_number}")
            all_pages, error = self.document_processor.GetDocAndPages()
            if error:
                document_extractor_logger.error(error)
                return None, None, error  # Return None for image paths and folder name, and error message
            if not all_pages:
                error_msg = "No pages found in PDF."
                document_extractor_logger.error(error_msg)
                return None, None, error_msg  # Return None for image paths and folder name, and error message

            selected_pages = [page_number] if page_number is not None else all_pages
            document_extractor_logger.info(f"Pages selected for processing: {selected_pages}")

            # Step 2: Extract images
            image_paths, unique_folder_name = self.document_processor.ExtractImageFromPdf(page_number, dpi=300)
            if not image_paths:
                error_msg = "Image extraction returned no results."
                document_extractor_logger.error(error_msg)
                return None, None, error_msg  # Return None for image paths and folder name, and error message

            document_extractor_logger.info("Images extracted and saved successfully.")
            return image_paths, unique_folder_name, None  # Return image paths, folder name, and no error

        except Exception as e:
            error_msg = f"Exception during document processing: {str(e)}"
            document_extractor_logger.error(error_msg)
            document_extractor_logger.error(traceback.format_exc())
            return None, None, error_msg  # Return None for image paths and folder name, and exception message

def preprocess_text(text):
    """
    Cleans and normalizes input text using basic NLP techniques.
    This function performs the following steps:
    1. Checks if the input text is provided; logs a warning and returns None if not.
    2. Normalizes whitespace and removes unwanted characters, retaining only alphanumeric characters and common punctuation.
    3. Splits the text into sentences using sentence tokenization.
    4. Tokenizes each sentence into words, converts them to lowercase, and filters out unwanted tokens, keeping only alphanumeric words and select punctuation.
    5. Joins the cleaned words back into sentences and combines them into the final cleaned text separated by newlines.
    6. Logs the progress and any issues encountered during preprocessing.
    Args:
        text (str): The input text to be preprocessed.
    Returns:
        str or None: The cleaned and normalized text, or None if preprocessing fails or input is invalid.
    """
    """Apply basic NLP cleaning and normalization."""
    try:
        if not text:
            document_extractor_logger.warning("No text provided for preprocessing.")
            return None
        
        document_extractor_logger.info("Starting text preprocessing.")
        document_extractor_logger.debug(f"Original extracted text length: {len(text)}")
        # Normalize whitespace and remove unwanted characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r"[^a-zA-Z0-9.,;:!?'\"]+", ' ', text)
        sentences = sent_tokenize(text)
        if not sentences:
            document_extractor_logger.warning("No sentences found after tokenization.")
            return None
        document_extractor_logger.debug(f"Total sentences after sentence tokenization: {len(sentences)}")

        cleaned_sentences = []
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            filtered_words = [
                word.lower() for word in words if word.isalnum() or word in [",", ".", ":", ";", "!", "?"]
            ]
            cleaned_sentence = " ".join(filtered_words)
            cleaned_sentences.append(cleaned_sentence)
            document_extractor_logger.debug(f"Cleaned sentence {i+1}: {cleaned_sentence}")
        if not cleaned_sentences:
            document_extractor_logger.warning("No cleaned sentences generated after filtering.")
            return None
        cleaned_text = "\n".join(cleaned_sentences)
        document_extractor_logger.info(f"Completed text preprocessing. Final text length: {len(cleaned_text)}")
        return cleaned_text

    except Exception as e:
        document_extractor_logger.error(f"Error during text preprocessing: {e}", exc_info=True)
        return None

def extract_text_from_file(file_path, file_type):
    """Extracts and preprocesses text based on the file type."""
    document_extractor_logger.info(f"Starting text extraction for: {file_path} (type: {file_type})")

    def extract_pdf():
        """
        Extracts text content from a PDF file specified by the global 'file_path' variable.

        Reads the PDF file, extracts text from each page, and concatenates the results.
        Logs debug information about the extraction process and warns if no text is found.

        Returns:
            str or None: The extracted text as a single string, or None if no text was extracted.
        """
        """Extract text from a PDF file."""
        document_extractor_logger.debug("Using PDF extractor")
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        if not text:
            document_extractor_logger.warning("No text extracted from PDF.")
            return None
        document_extractor_logger.debug(f"Extracted {len(reader.pages)} pages from PDF")
        return text
    
    def extract_docx():
        """
        Extracts text content from a DOCX file specified by the global `file_path` variable.

        Uses the `docx2txt` library to process the DOCX file and logs the extraction process.
        If no text is extracted, logs a warning and returns None. Otherwise, returns the extracted text.

        Returns:
            str or None: The extracted text from the DOCX file, or None if extraction fails.
        """
        document_extractor_logger.debug("Using DOCX extractor")
        text = docx2txt.process(file_path)
        if not text:
            document_extractor_logger.warning("No text extracted from DOCX.")
            return None
        document_extractor_logger.debug(f"Extracted text length from DOCX: {len(text)}")
        return text
    
    def extract_doc():
        """
        Extracts text content from a DOC file using the Microsoft Word COM interface.

        This function opens a DOC file specified by the global variable `file_path`, extracts its text content,
        and returns it as a string. It logs the extraction process and handles cases where no text is found.

        Returns:
            str or None: The extracted text from the DOC file, or None if no text was extracted.

        Raises:
            Exception: If there is an error opening or reading the DOC file.
        """
        document_extractor_logger.debug("Using DOC extractor")
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(file_path)
        text = doc.Content.Text
        doc.Close()
        word.Quit()
        if not text:
            document_extractor_logger.warning("No text extracted from DOC.")
            return None
        document_extractor_logger.debug(f"Extracted text length from DOC: {len(text)}")
        return text
    
    def extract_html():
        """
        Extracts and returns the text content from an HTML file.

        Reads the HTML file specified by `file_path`, parses it using BeautifulSoup,
        and extracts all textual content, separating lines with newline characters.
        Logs debug information about the extraction process and warns if no text is found.

        Returns:
            str or None: The extracted text if successful, or None if no text is found.
        """
        document_extractor_logger.debug("Using HTML extractor")
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            text = soup.get_text(separator="\n")
        if not text:
            document_extractor_logger.warning("No text extracted from HTML.")
            return None
        document_extractor_logger.debug(f"Extracted text length from HTML: {len(text)}")
        return text
    
    def extract_txt():
        """
        Extracts text content from a TXT file specified by the global variable `file_path`.

        Reads the file using UTF-8 encoding and returns its content as a string.
        Logs debug messages about the extraction process and warns if no text is extracted.

        Returns:
            str or None: The extracted text if successful, or None if the file is empty.
        """
        document_extractor_logger.debug("Using TXT extractor")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text:
            document_extractor_logger.warning("No text extracted from TXT.")
            return None
        document_extractor_logger.debug(f"Extracted text length from TXT: {len(text)}")
        return text
        
    def read_excel():
        """
        Reads an Excel file and extracts a summary of its contents.
        This function reads up to 100 rows from the specified Excel file, generates a summary including dataset info,
        descriptive statistics, and a sample of the data in Markdown format. It logs relevant debug and warning messages
        during the extraction process.
        Returns:
            str or None: A Markdown-formatted string containing dataset info, description, and sample data,
            or None if the file is empty or no text could be extracted.
        """
        document_extractor_logger.debug("Using Excel extractor")
        df = pd.read_excel(file_path, nrows=100)
        if df.empty:
            document_extractor_logger.warning("No data extracted from Excel file.")
            return None
        buffer = StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        desc_str = df.describe(include='all').to_markdown()
        markdown_table = df.to_markdown(index=False)
        text = f"### Dataset Info:\n{info_str}\n\n### Dataset Description:\n{desc_str}\n\n### Sample Data:\n{markdown_table}"
        if not text:
            document_extractor_logger.warning("No text extracted from Excel file.")
            return None
        document_extractor_logger.debug(f"Extracted summary from Excel file with shape {df.shape}")
        return text
      
    extractors = {
        "pdf": extract_pdf,
        "docx": extract_docx,
        "doc": extract_doc,
        "html": extract_html,
        "txt": extract_txt,
        "xlsx": read_excel
    }

    try:
        document_extractor_logger.info(f"Extracting text from {file_type} file: {file_path}")
        if not file_type:
            document_extractor_logger.error("File type is not provided.")
            raise ValueError("File type is not provided.")
        extractor = extractors.get(file_type)
        if not extractor:
            document_extractor_logger.warning(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")

        raw_text = extractor()
        if raw_text is None:
            document_extractor_logger.error(f"Failed to extract text from {file_type} file: {file_path}")
            raise RuntimeError(f"Failed to extract text from {file_type} file: {file_path}")
        document_extractor_logger.info(f"Successfully extracted text from {file_type} file: {file_path}")
        
        cleaned_text = preprocess_text(raw_text)
        if cleaned_text:
            document_extractor_logger.info("Text preprocessing completed successfully.")
        else:
            document_extractor_logger.warning("Text preprocessing returned None.")
        return cleaned_text

    except Exception as e:
        document_extractor_logger.error(f"Failed to extract text from {file_type} file: {e}", exc_info=True)
        raise RuntimeError(f"Failed to extract text from {file_type} file: {str(e)}")
