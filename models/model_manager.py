# models/model_manager.py
from core.llama_model import LlamaSummarizer
from utils.config import Config

loggers = Config.init_logging()
service_logger = loggers['model_manager']

class ModelManager:
    models = {}

    @staticmethod
    def get_model(key):
        if key not in ModelManager.models:
            try:
                if key == "llama":
                    model = LlamaSummarizer()
                    ModelManager.models[key] = model
                    service_logger.info("LLaMA summarizer loaded successfully.")
                else:
                    error_message = f"Unknown model key: {key}"
                    service_logger.error(error_message)
                    raise ValueError(error_message)
            except Exception as e:
                service_logger.exception(f"Failed to load model '{key}': {str(e)}")
                return None, str(e)
        else:
            service_logger.debug(f"Model '{key}' retrieved from cache.")
        return ModelManager.models.get(key), None
