from models.model_manager import ModelManager

def load_llama_model():
    """
    Loads the 'llama' model using the ModelManager.
    Returns:
        tuple: A tuple containing the loaded model (or None if loading failed) and an error object (or None if successful).
    """
    model, error = ModelManager.get_model("llama")
    return model,error
    