import requests
import base64
import json
import urllib3

# url = "https://llm-fastapi-sreejith8100.hf.space/predict"
url = "https://sreejith8100-llm-fastapi.hf.space/predict"

# Read image bytes and encode as base64 string
with open(r"", "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

payload = {
    "inputs": {
        "image": image_base64,
        "question": "What is in the image?",
        "stream": True
    }
}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

headers = {
    "Content-Type": "application/json"
}

try:
    with requests.post(url, data=json.dumps(payload), headers=headers, stream=True, verify=False) as response:
        response.raise_for_status()  # Raise error for HTTP errors

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    content = decoded_line.replace("data: ", "")
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Malformed JSON chunk: {content}") from e

                    output = parsed.get("output", "")
                    if output:
                        print(output, end="", flush=True)

except requests.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
    raise

except Exception as err:
    print(f"Error occurred: {err}")
    raise
