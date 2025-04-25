# Start from CUDA 12.1 base with build tools
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Install Miniconda (if not already part of the base image)
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y zlib1g-dev

# Copy environment and source code   
COPY . .

# Install Python dependencies using pip
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clone and install RiNALMo within the environment
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c "cd /app/rnachat/rinalmo && \
    pip install . && \
    pip install flash-attn==2.3.2"


# Expose port
EXPOSE 7860

RUN pip list | grep rinalmo


# Entry point
# CMD ["python", "train.py"]
CMD ["/bin/bash", "-c", "sleep 3600 && python train.py"]
