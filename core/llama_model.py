import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,TextStreamer
from utils.config import Config
from transformers import TextIteratorStreamer
import threading
import requests
import os
import json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

loggers = Config.init_logging()
llama_logger = loggers['llama_model']

USE_SERVER = Config.USE_SERVER
API_URL_LLAMA = Config.API_URL_LLAMA
HEADER_TOKEN = Config.HEADER_TOKEN

class LlamaSummarizer:
    _instance = None

    def __new__(cls, model_id=r"Weights\LLAMA3\llama-3.2-3B-Instruct"):
        if cls._instance is None:
            cls._instance = super(LlamaSummarizer, cls).__new__(cls)
            cls._instance.model_id = model_id
            cls._instance.use_server = USE_SERVER
            cls._instance.HF_TOKEN = HEADER_TOKEN
            cls._instance.api_url=API_URL_LLAMA
            try:
                if not USE_SERVER:
                    llama_logger.info(f"Loading tokenizer and model: {model_id}")
                    cls._instance.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    cls._instance.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    )
                    cls._instance.streamer = TextStreamer(cls._instance.tokenizer, skip_prompt=True)
                    cls._instance.pipe = pipeline(
                        "text-generation",
                        model=cls._instance.model,
                        tokenizer=cls._instance.tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        streamer=cls._instance.streamer
                    )
                    llama_logger.info(f"LLaMA model loaded successfully: {model_id}")
                else:
                    llama_logger.info("Using Hugging Face Inference API endpoint")
            except Exception as e:
                llama_logger.error(f"Failed to load LLaMA model: {e}")
                raise
            cls._instance.input_token_count = 0
            cls._instance.generated_token_count = 0
        return cls._instance

    def format_messages(self, messages):
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        return prompt + "<|assistant|>\n"

    def get_token_counts(self):
        return {
            "input_tokens": self.input_token_count,
            "generated_tokens": self.generated_token_count,
            "total_tokens": self.input_token_count + self.generated_token_count
        }
    
    def load_history(self, query_id):
        path = f"./history/{query_id}.json"
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    llama_logger.info(f"[HISTORY] Loaded {len(data)} messages from history.")
                    return data, None
            except Exception as e:
                llama_logger.error(f"[HISTORY LOAD ERROR] {e}")
                return [], f"Failed to load history: {str(e)}"
        else:
            llama_logger.info("[HISTORY] No existing history found.")
            return [], None

    def save_history(self, query_id, messages):
        path = f"./history/{query_id}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(messages, f, indent=2)
            llama_logger.info(f"[HISTORY] Saved {len(messages)} messages to history.")
            return True, None
        except Exception as e:
            llama_logger.error(f"[HISTORY SAVE ERROR] {e}")
            return False, str(e)
    
    def summarize_via_hf_api(self, messages, temperature=0.7, max_new_tokens=512):
        prompt = self.format_messages(messages)
        headers = {
            "Authorization": f"Bearer {self.HF_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "return_full_text": False
            },
            "options": {
                "use_cache": True,
                "wait_for_model": True
            }
        }
        try:
            llama_logger.info(f"Calling Hugging Face Inference API at {self.api_url}")
            llama_logger.info(f"{payload}")
            response = requests.post(self.api_url, headers=headers, json=payload,verify=False)
            response.raise_for_status()
            output = response.json()
            return output[0]["generated_text"]
        except Exception as e:
            llama_logger.error(f" HF API call failed: {e}")
            return f" Error: {e}"

    def stream_summary(self, max_new_tokens, temperature, messages,query_id=None,use_history=False):
        llama_logger.info("Starting streaming summarization request.")
        llama_logger.info(f"Query ID: {query_id}")
        llama_logger.info(f"send max_new_tokens: {max_new_tokens}")
        llama_logger.info(f"temperature: {temperature}")
        llama_logger.info(f"messages: {messages}")
        llama_logger.info(f"Use history: {use_history}")

        if query_id:
            self.query_id = query_id
            history, _ = self.load_history(query_id) if use_history else ([], None)
            msgs = history + messages
        else:
            msgs = messages

        formatted_prompt = self.format_messages(msgs)
        llama_logger.info(f"Formatted prompt: {formatted_prompt}")
        if self.use_server:
            result = self.summarize_via_hf_api(msgs, temperature, max_new_tokens)
            yield result
            return

        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.model.device)
        self.input_token_count = input_ids.shape[-1] 
        max_context_length = 22000
        if self.input_token_count + max_new_tokens > max_context_length:
            error_msg = f"Input is too long ({self.input_token_count} tokens). Exceeds context window of {max_context_length} with max_new_tokens={max_new_tokens}."
            llama_logger.error(error_msg)
            yield {
                "error": "ContextLimitExceeded",
                "input_tokens": self.input_token_count,
                "max_allowed": max_context_length - max_new_tokens,
                "message": error_msg
            }
            return
        try:
            # Set up the streamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            # Launch generation in a background thread
            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": temperature,
                "streamer": streamer
            }
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            generated_text = ""
            for new_text in streamer:
                self.generated_token_count += len(self.tokenizer.encode(new_text)) 
                generated_text += new_text
                yield new_text

            if query_id and use_history:
                # Append bot reply to history
                llama_logger.info(f"Appending bot reply to history.{generated_text}")
                msgs.append({"role": "assistant", "content": generated_text})
                self.save_history(query_id, msgs)

        except Exception as e:
            llama_logger.error(f"Streaming summarization failed: {e}")
            yield {
                "error": "StreamingFailed",
                "message": str(e)
            }