from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import types
import json
from pydantic import validator
from endpoint_handler import EndpointHandler  # your handler file
import base64

app = FastAPI()

handler = None

@app.on_event("startup")
async def load_handler():
    global handler
    handler = EndpointHandler()

class PredictInput(BaseModel):
    image: str       # base64-encoded image string
    question: str
    stream: bool = False

    @validator("question")
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question must not be empty")
        return v

    @validator("image")
    def valid_base64_and_size(cls, v):
        try:
            decoded = base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("`image` must be valid base64")
        if len(decoded) > 10 * 1024 * 1024:  # 10 MB limit
            raise ValueError("Image exceeds 10 MB after decoding")
        return v

class PredictRequest(BaseModel):
    inputs: PredictInput

@app.get("/")
async def root():
    return {"message": "FastAPI app is running on Hugging Face"}

@app.post("/predict")
async def predict_endpoint(payload: PredictRequest):
    """
    Handles prediction requests by processing the input payload and returning the prediction result.
    Args:
        payload (PredictRequest): The request payload containing the input data for prediction, including image, question, and stream flag.
    Returns:
        JSONResponse: If a ValueError occurs, returns a JSON response with an error message and status code 400.
        JSONResponse: If any other exception occurs, returns a JSON response with a generic error message and status code 500.
        StreamingResponse: If the prediction result is a generator (streaming), returns a streaming response with event-stream media type, yielding prediction chunks as JSON.
    Notes:
        - Logs the received question for debugging purposes.
        - Handles both standard and streaming prediction results.
        - Structured JSON messages are sent to indicate the end of the stream or errors during streaming.
    """
    print(f"[Request] Received question: {payload.inputs.question}")
    
    data = {
        "inputs": {
            "image": payload.inputs.image,
            "question": payload.inputs.question,
            "stream": payload.inputs.stream
        }
    }
    
    try:
        result = handler.predict(data)
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": "Internal server error"}, status_code=500)

    if isinstance(result, types.GeneratorType):
        def event_stream():
            try:
                for chunk in result:
                    yield f"data: {json.dumps(chunk)}\n\n"
                # Return structured JSON to indicate end of stream
                yield f"data: {json.dumps({'end': True})}\n\n"
            except Exception as e:
                # Return structured JSON to indicate error
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

