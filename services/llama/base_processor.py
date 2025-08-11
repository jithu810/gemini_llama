import traceback
from utils.config import Config
from utils.value_check import safe_int, safe_float
from utils.response_utils import log_and_respond
from utils.utils import cleanup_folder
from validators.validation_helpers import validate_pdf_input
from core.llama_model import LlamaSummarizer

loggers = Config.init_logging()

TEMPERATURE = Config.TEMPERATURE
MAX_NEW_TOKENS = Config.MAX_NEW_TOKENS

class BaseDocumentProcessor:
    def __init__(self, params: dict, context):
        self.params: dict = params
        self.context = context
        self.logger = loggers

        try:
            self.doc_path: str = params.get("FilePath")
            self.query_id: str = params.get("QueryId")

            self.temperature: str = safe_float(params.get("temperature", TEMPERATURE), TEMPERATURE)
            self.max_new_tokens: str = safe_int(params.get("max_new_tokens", MAX_NEW_TOKENS), MAX_NEW_TOKENS)

            self.llama_model = LlamaSummarizer()

            self.respond = log_and_respond
            self.clean = cleanup_folder
            self.validate_input = validate_pdf_input

        except Exception as e:
            module = self.context.get("module", "default")
            log = self.logger.get(module) or self.logger["default"]
            log.error(f"[BASE CLASS ERROR] Failed to initialize BaseDocumentProcessor: {e}")
            log.debug(traceback.format_exc())
            raise

    def process(self):
        """Call this method in your handler"""
        # try:
        #     # Example validation
        #     if not self.validate_input(self.pdf_path, self.page_number):
        #         return self.respond(
        #             success=False,
        #             query_id=self.query_id,
        #             message="Invalid PDF input.",
        #             logger=self.logger.get(self.context.get("module", "default"))
        #         )

        #     # Main AI processing call
        #     result = self.ai_processor.process(
        #         pdf_path=self.pdf_path,
        #         page_number=self.page_number,
        #         temperature=self.temperature,
        #         max_new_tokens=self.max_new_tokens,
        #         query_id=self.query_id
        #     )

        #     return self.respond(
        #         success=True,
        #         query_id=self.query_id,
        #         message="Processing successful.",
        #         data=result,
        #         logger=self.logger.get(self.context.get("module", "default"))
        #     )

        # except Exception as e:
        #     module = self.context.get("module", "default")
        #     log = self.logger.get(module) or self.logger["default"]
        #     log.error(f"[PROCESS_ERROR] Error while processing document: {e}")
        #     log.debug(traceback.format_exc())

        #     return self.respond(
        #         success=False,
        #         query_id=self.query_id,
        #         message=f"Exception occurred during processing: {str(e)}",
        #         logger=log
        #     )
