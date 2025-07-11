from PIL import Image
import fitz
import os
from utils.config import Config
import uuid
loggers = Config.init_logging()
pdf_loader = loggers['pdf_loader']

class PDFHandler:
    def __init__(self, pdf_path):
        """
        Initialize the PDFHandler with the path to the PDF file.
        Args:
            pdf_path (str): The file path to the PDF document.
        """
        self.pdf_path = pdf_path
        self.pdf_buffer_memmory=None
        self.pdflen=None
        pdf_loader.info(f"PDFHandler initialized with PDF path: {pdf_path}")

    def load_pdf(self):
        """
        Load the PDF document from the specified file path.
        Returns:
            fitz.Document: The loaded PDF document object.
        """
        try:
            pdf_loader.info("Loading PDF document.")
            self.pdf_buffer_memmory=fitz.open(self.pdf_path)
            self.pdflen = list(range(len(self.pdf_buffer_memmory)))
            return self.pdflen
        except Exception as e:
            pdf_loader.error(f"Error while retrieving pages: {e}")
            return None  

    def extract_images_from_pages(self,selected_pages,output_dir,dpi=300):
        """
        Extract images from the specified pages of the PDF with high resolution.

        Args:
            output_dir (str): The folder where images will be saved.
            selected_pages (list of int): The list of page numbers to extract images from.
            dpi (int): The resolution for image extraction. Default is 300.

        Returns:
            list of str: A list of file paths to the saved images.
        """
        try:
            pdf_loader.info(f"Extracting images from pages: {selected_pages} with DPI: {dpi}")
            image_paths = []
            if self.pdf_buffer_memmory is None:
                pdf_loader.error("PDF not loaded. Cannot extract images.")
                return []
            
            # Create a unique folder
            unique_folder_name = f"images_{uuid.uuid4().hex}"
            unique_folder_path = os.path.join(output_dir, unique_folder_name)
            os.makedirs(unique_folder_path, exist_ok=True)

            for page_number in selected_pages:
                try:
                    if 0 <= page_number < len(self.pdf_buffer_memmory):
                        page = self.pdf_buffer_memmory.load_page(page_number)
                        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                        img = Image.frombytes("RGB",[pix.width, pix.height],pix.samples)
                        pdf_loader.info(f"Extracting image from page {page_number + 1} with size {pix.width}x{pix.height} at {dpi} DPI.")
                        image_path = os.path.join(unique_folder_path, f'page_{page_number + 1}.png')
                        img.save(image_path)
                        image_paths.append(image_path)
                        # pdf_loader.info(f"Image from page {page_number} saved successfully to {image_path}.")
                    else:
                        pdf_loader.warning(f"Page number {page_number} is out of range.")
                except Exception as e:
                    pdf_loader.error(f"Error extracting image from page {page_number}: {e}")
            self.pdf_buffer_memmory.close()
            return image_paths,unique_folder_name
        except Exception as e:
            pdf_loader.error(f"Unexpected error while extracting images: {e}") 
            return []
