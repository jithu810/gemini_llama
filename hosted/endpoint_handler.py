import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from io import BytesIO
import base64
from huggingface_hub import login
from huggingface_hub import login
import os

class EndpointHandler:
    def __init__(self, model_dir=None):
        print("[Init] Initializing EndpointHandler...")
        self.load_model()

    def load_model(self):
        hf_token = os.getenv("HF_TOKEN")
        model_path = "openbmb/MiniCPM-o-2_6"  # use model repo name directly

        if hf_token:
            print("[Auth] Logging into Hugging Face Hub with token...")
            login(token=hf_token)

        print(f"[Model Load] Loading model from: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype='auto',  # safer on Spaces
                init_vision=True,
                init_audio=False,
                init_tts=False
            ).eval().cuda()
            print("[Model Load] Model successfully loaded and moved to CUDA.")
        except Exception as e:
            print(f"[Model Load Error] {e}")
            raise RuntimeError(f"Failed to load model: {e}")
        
    def load_image(self, image_base64):
        try:
            print("[Image Load] Decoding base64 image...")
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            print("[Image Load] Image successfully decoded and converted to RGB.")
            return image
        except Exception as e:
            print(f"[Image Load Error] {e}")
            raise ValueError(f"Failed to open image from base64 string: {e}")

    def predict(self, request):
        print(f"[Predict] Received request: {request}")

        image_base64 = request.get("inputs", {}).get("image")
        question = request.get("inputs", {}).get("question")
        stream = request.get("inputs", {}).get("stream", False)

        if not image_base64 or not question:
            print("[Predict Error] Missing 'image' or 'question' in the request.")
            return {"error": "Missing 'image' or 'question' in inputs."}

        try:
            image = self.load_image(image_base64)
            msgs = [{"role": "user", "content": [image, question]}]

            print(f"[Predict] Asking model with question: {question}")
            print("[Predict] Starting chat inference...")

            res = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                stream=stream
            )

            if stream:
                for new_text in res:
                    yield {"output": new_text}
            else:
                generated_text = "".join(res)
                print("[Predict] Inference complete.")
                return {"output": generated_text}

        except Exception as e:
            print(f"[Predict Error] {e}")
            return {"error": str(e)}

    def __call__(self, data):
        print("[__call__] Invoked handler with data.")
        return self.predict(data)
