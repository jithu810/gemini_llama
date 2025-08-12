from utils.config import Config
from transformers import TextIteratorStreamer
import threading
import requests
import os
import json
from core.llama.loaders import llama_3b,load_model_V_4_0,llama_1b
from core.llama.history import ChatHistoryManager
chat_history=ChatHistoryManager()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

loggers = Config.init_logging()
llama_logger = loggers['llama_model']

USE_SERVER = Config.USE_SERVER
API_URL_LLAMA = Config.API_URL_LLAMA
HEADER_TOKEN = Config.HEADER_TOKEN
MODEL_VERSION = Config.MODEL_VERSION

class LlamaSummarizer:
    _instance = None
    version = MODEL_VERSION

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaSummarizer, cls).__new__(cls)
            cls._instance.model_name = MODEL_VERSION
            cls._instance.use_server = USE_SERVER
            cls._instance.query_id = None
            cls._instance.version = MODEL_VERSION.lower()
            if cls._instance.use_server:
                llama_logger.info("Using Hugging Face Inference API endpoint")
            else:
                cls._instance.model, cls._instance.tokenizer = cls._instance.load_model_and_tokenizer()
                llama_logger.info(f"MiniCPMModel initialized with model: {cls._instance.model_name}")
            cls._instance.input_token_count = 0
            cls._instance.generated_token_count = 0
        return cls._instance
        
    def load_model_and_tokenizer(self):
        """"""
        version = self.version
        if version == "llama_1b":
            return llama_1b()
        elif version == "llama_3b":
            return llama_3b()
        elif version == "v_4":
            return load_model_V_4_0()
        else:
            raise ValueError(f"Unsupported MiniCPM version: {version}")

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
            history, _ = chat_history.load_history(query_id) if use_history else ([], None)
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
                chat_history.save_history(query_id, msgs)

        except Exception as e:
            llama_logger.error(f"Streaming summarization failed: {e}")
            yield {
                "error": "StreamingFailed",
                "message": str(e)
            }