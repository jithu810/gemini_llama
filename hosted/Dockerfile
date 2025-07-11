FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y wget
RUN useradd -m -u 1000 user

USER user
WORKDIR /app

ENV PATH="/home/user/.local/bin:$PATH"
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface
ENV TORCH_CUDA_ARCH_LIST="8.0+PTX"

RUN wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.4/flash_attn-2.7.3+cu121torch2.3-cp310-cp310-linux_x86_64.whl
RUN pip install ./flash_attn-2.7.3+cu121torch2.3-cp310-cp310-linux_x86_64.whl && rm flash_attn-2.7.3+cu121torch2.3-cp310-cp310-linux_x86_64.whl

COPY --chown=user requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
