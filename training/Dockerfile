FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /app

# System deps + Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends git python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch + Unsloth from scratch (no conflicting base image versions)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
        "unsloth @ git+https://github.com/unslothai/unsloth.git" \
        xformers && \
    pip install --no-cache-dir --no-deps \
        trl peft accelerate bitsandbytes && \
    pip install --no-cache-dir \
        openenv-core datasets

# Copy project code
COPY . /app/stack_doctor/
ENV PYTHONPATH="/app/stack_doctor:$PYTHONPATH"

CMD ["python", "/app/stack_doctor/training/train_stack_doctor.py"]
