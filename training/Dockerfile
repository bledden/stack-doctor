FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install Unsloth + explicitly fix torch/torchvision compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "unsloth @ git+https://github.com/unslothai/unsloth.git" \
        xformers && \
    pip install --no-cache-dir "torchvision>=0.25.0" && \
    pip install --no-cache-dir --no-deps \
        trl \
        peft \
        accelerate \
        bitsandbytes

# Verify unsloth imports before proceeding
RUN python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"

# Install openenv for environment stepping during reward computation
RUN pip install --no-cache-dir openenv-core datasets

# Copy project code
COPY . /app/stack_doctor/
ENV PYTHONPATH="/app/stack_doctor:$PYTHONPATH"

# Default: run training
CMD ["python", "/app/stack_doctor/training/train_stack_doctor.py"]
