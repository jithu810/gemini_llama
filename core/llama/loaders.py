from transformers import AutoModel, AutoTokenizer,AutoConfig
import torch
from utils.config import Config
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,TextStreamer

minicpm_model = Config.init_logging()['minicpm_model']

def llama_1b():
    try:
        model_path = r"Weights\Llama-3.2-1B-Instruct"
        minicpm_model.info("Loading quantized MiniCPM-V2 model")
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

def load_model_o_2_6():
    try:
        model_path = r"Weights\MiniCPM-o-2_6"
        minicpm_model.info("Loading MiniCPM-O-2_6 model")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            init_vision=True,
            init_audio=False,
            init_tts=False
        ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        minicpm_model.info("O_2_6 model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        minicpm_model.error(f"Error loading o_2_6 model: {e}", exc_info=True)
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

