
import os
import time
import base64
import requests
import json
from utils.config import Config

loggers = Config.init_logging()
minicpm_model = loggers['minicpm_model']

HEADER_TOKEN = Config.HEADER_TOKEN

def extract_text_via_hf_api_base_stream(img_path, question, api_url, verify_ssl=False,stream=False):
    timing = {}
    # 1) Read & size‐check
    try:
        minicpm_model.info("Reading image bytes from file.")
        with open(img_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        minicpm_model.error(f"Image file not found: {img_path}")
        raise

    MAX_IMG_BYTES = 10 * 1024 * 1024
    if len(image_bytes) > MAX_IMG_BYTES:
        raise ValueError(f"Image exceeds {MAX_IMG_BYTES/(1024*1024):.1f} MB")

    # 2) Base64 encode
    try:
        minicpm_model.info("Encoding image to base64.")
        start_enc = time.time()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        timing["base64_encode"] = time.time() - start_enc
    except Exception as e:
        minicpm_model.error(f"Failed to base64 encode: {e}")
        raise

    # 3) Build payload
    minicpm_model.info("Preparing payload for Hugging Face API.")
    start_pl = time.time()
    payload = {
        "inputs": {
            "image": base64_image,
            "question": question,
            "stream": stream
        },
        "parameters": {}
    }
    timing["prepare_payload"] = time.time() - start_pl

    # 4) HTTP POST with explicit read‐timeout
    minicpm_model.info("Sending streaming request to Hugging Face API.")
    headers = {
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {HEADER_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        req_start = time.time()
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            stream=stream,
            timeout=(10, 60),  # 10 s connect, 60 s per‐chunk read
            verify=verify_ssl
        )
        response.raise_for_status()
        timing["api_request"] = time.time() - req_start
    except requests.exceptions.HTTPError as he:
        minicpm_model.error(f"HTTP error: {he} (status {he.response.status_code})")
        yield json.dumps({"error": str(he)})
        return
    except requests.exceptions.RequestException as re:
        minicpm_model.error(f"Request failed: {re}")
        yield json.dumps({"error": str(re)})
        return
        # 5) Stream SSE lines
    minicpm_model.info("Streaming response received, iterating SSE lines.")
    try:
        for raw_line in response.iter_lines(chunk_size=1, decode_unicode=False):
            if raw_line is None:
                break
            decoded_line = raw_line.decode("utf-8", errors="ignore")
            if decoded_line.startswith("data: "):
                content = decoded_line[len("data: "):]
                minicpm_model.debug(f"Streamed chunk: {content}")
                yield content
        minicpm_model.info("Streaming from HF API completed successfully.")
    except requests.exceptions.ReadTimeout as rt:
        minicpm_model.error(f"Read timeout: {rt}")
        yield json.dumps({"error": "Read timeout from HF API"})
        return
    except requests.RequestException as e:
        minicpm_model.error(f"Error during streaming iteration: {e}")
        yield json.dumps({"error": str(e)})
        return
    finally:
        response.close()

    # 6) Final JSON “end” marker
    yield json.dumps({"end": True, "timing": timing})
    minicpm_model.info(f"Timing: {timing}")

