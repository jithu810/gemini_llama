# handlers/service_registry.py
from services.chromadb.chroma_db_insert_service import ChromaInsertProcessor
from services.chromadb.chroma_db_query_service import ChromaQueryProcessor

from services.llama.chat_service import ChatProcessor 
from services.llama.document_summary_service_convo import DocumentSummaryConvoProcessor
from services.llama.document_summary_service import DocumentSummaryProcessor
from services.llama.email_refine_service import EmailRefineProcessor
from services.llama.excel_column_comparison_service import ExcelColumnCompareProcessor
from services.llama.vessel_website_service import VesselWebsiteProcessor

from utils.config import Config
loggers= Config.init_logging()

logger = loggers['handler']

class ServiceRegistry:
    """
    ServiceRegistry is responsible for managing and executing various service processors based on a given method name.
    Attributes:
        services (dict): A mapping of method names (str) to their corresponding processor classes.
    Methods:
        __init__():
            Initializes the ServiceRegistry with a predefined mapping of service names to processor classes.
        execute(method_name: str, param_data: dict, context):
            Executes the processor associated with the given method name.
            Args:
                method_name (str): The name of the service method to execute.
                param_data (dict): The parameters to pass to the processor.
                context: Additional context required by the processor.
            Returns:
                The result returned by the processor's process() method.
            Raises:
                Exception: If the provided method_name does not exist in the services mapping.
    """
    def __init__(self):
        self.services = {
            "document_summary": DocumentSummaryProcessor,
            "chromadb_insert": ChromaInsertProcessor,
            "chromadb_query":ChromaQueryProcessor,  
            "chat": ChatProcessor,
            "document_summary_convo": DocumentSummaryConvoProcessor,
            "emailrefine":EmailRefineProcessor,
            "excel_column_compare":ExcelColumnCompareProcessor,
            "vessel_website":VesselWebsiteProcessor,
        }

    def execute(self, method_name: str, param_data: dict, context):
        """
        Executes a registered service method with the provided parameters and context.
        Args:
            method_name (str): The name of the service method to execute.
            param_data (dict): A dictionary containing parameters required by the service method.
            context: An object providing contextual information for the service execution.
        Returns:
            Any: The result returned by the service processor's `process` method.
        Raises:
            Exception: If the specified method_name is not registered in the services.
        """
        logger.debug(f"Executing service for method: {method_name}")
        logger.debug(f"Received param_data: {param_data}")

        if method_name not in self.services:
            logger.error(f"Unknown method: {method_name}")
            raise Exception(f"Unknown method: {method_name}") 
        
        processor_class = self.services[method_name]
        logger.debug(f"Using processor class: {processor_class.__name__}")
        processor = processor_class(param_data, context)
        result = processor.process()
        logger.debug(f"Result from processor: {result}")
        return result
    