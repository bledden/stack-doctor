FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Step 1: Install unsloth first (it pulls the torch version it wants)
RUN pip install --no-cache-dir \
    "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    unsloth_zoo \
    xformers

# Step 2: Now install torchvision to match whatever torch unsloth installed
RUN pip install --no-cache-dir --upgrade torchvision

# Step 3: Install TRL + training deps
RUN pip install --no-cache-dir \
    "trl>=0.18.2,<=0.24.0" \
    "peft>=0.18.0" \
    "accelerate>=0.34.1" \
    "bitsandbytes>=0.45.5" \
    "datasets>=3.4.1" \
    "transformers>=4.51.3" \
    "huggingface_hub>=0.34.0" \
    sentencepiece \
    hf_transfer \
    torchao \
    triton

# Step 4: Install openenv for environment stepping
RUN pip install --no-cache-dir openenv-core

# Copy project code
COPY . /app/stack_doctor/
ENV PYTHONPATH="/app/stack_doctor:$PYTHONPATH"

CMD ["python", "/app/stack_doctor/training/train_stack_doctor.py"]
