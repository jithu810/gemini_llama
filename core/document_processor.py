# -*- coding: utf-8 -*-
from .pdf_handler import PDFHandler
from utils.config import Config
loggers = Config.init_logging()

document_extractor_logger = loggers['document_extractor']

class DocumentProcessor:
    def __init__(self, pdf_path, output_dir):
        """
        Initialize the InvoiceProcessor with the path to the PDF and the output directory.

        Args:
            pdf_path (str): The file path to the PDF document.
            output_dir (str): The directory where extracted text files will be saved.
        """
        self.pdf_handler = PDFHandler(pdf_path)
        self.pdf_path=pdf_path
        self.output_dir = output_dir
        document_extractor_logger.info(f"InvoiceProcessor initialized with pdf path '{pdf_path}' andoutput directory: '{output_dir}' ")

    def GetDocAndPages(self):
        """
        Get a list of all page numbers in the PDF.
        Returns:
            tuple: (list of int, None) on success, or (None, error_message) on failure.
        """
        try:
            pdflen = self.pdf_handler.load_pdf()
            return pdflen, None
        except Exception as e:
            error_msg = f"Error while retrieving pages: {e}"
            document_extractor_logger.error(error_msg)
            return None, error_msg

    def ExtractImageFromPdf(self, page_number, dpi=200):
        """
        Extract images from the specified pages of the PDF.

        Args:
            page_number (int, optional): The specific page number to extract images from.
                                        If None, extract images from all pages.
        
        Returns:
            list of str: The list of image paths that were processed.
        """
        try:
            document_extractor_logger.info(f"Starting to convert PDF to image with selected page,{page_number}")
            if page_number is not None:
                document_extractor_logger.info(f"User selects custom pages ,{page_number}")
                selected_pages = [page_number]
            else:
                document_extractor_logger.info(f"User dosenot selects any pages,{page_number}")
                selected_pages = self.pdf_handler.pdflen 

            document_extractor_logger.info(f"selected pages for extract_images_from_pages are,{selected_pages}")
            image_paths,unique_folder_name = self.pdf_handler.extract_images_from_pages(selected_pages,self.output_dir,dpi)
            if image_paths:
                document_extractor_logger.info(f"Number of pages processed: {len(image_paths)}")
            else:
                document_extractor_logger.warning("No images were extracted.")
            return image_paths,unique_folder_name

        except Exception as e:
            document_extractor_logger.error(f"Error during PDF to image conversion: {e}")
            return []  

        