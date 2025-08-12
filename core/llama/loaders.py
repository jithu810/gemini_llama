from transformers import AutoModel, AutoTokenizer,AutoConfig
import torch
from utils.config import Config
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,TextStreamer

minicpm_model = Config.init_logging()['minicpm_model']

def llama_1b():
    try:
        model_path = r"Weights\Llama-3.2-1B-Instruct"
        minicpm_model.info(f"Loading model from path{model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        streamer = TextStreamer(tokenizer,skip_prompt=True)
        return model,tokenizer
    except Exception as e:
        minicpm_model.error(f"Error loading V2_6_int4 model: {e}", exc_info=True)
        return None, None

def llama_3b():
    try:
        model_path = r"Weights\LLAMA3\llama-3.2-3B-Instruct"
        minicpm_model.info(f"Loading model from path{model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        streamer = TextStreamer(tokenizer,skip_prompt=True)
        return model,tokenizer
    except Exception as e:
        minicpm_model.error(f"Error loading V2_6_int4 model: {e}", exc_info=True)
        return None, None


def load_model_V_4_0():
    try:
        minicpm_model.info("Loading MiniCPM-V4 model")
        model_path = r"D:\Main_releases\Vision_model\Weights\MiniCPM-V-4"
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                  attn_implementation='sdpa', torch_dtype=torch.bfloat16).eval().cuda()
        print("model loaded sucessfully")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("tokenizer loaded sucessfully")
        minicpm_model.info("V4 model and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        minicpm_model.error(f"Error loading V4 model: {e}", exc_info=True)
        return None, None

