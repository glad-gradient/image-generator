FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Install any necessary dependencies
RUN apt-get update && apt-get install -y \
wget

RUN wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image.py -P app/

# ENV LD_LIBRARY_PATH="/usr/lib/wsl/lib/"
RUN pip3 uninstall torch torchvision -y
RUN pip uninstall bitsandbytes

RUN pip install --upgrade pip
RUN pip install python==3.8
COPY requirements.txt /app/
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r /app/requirements.txt

# RUN accelerate config default # Sets the device to CPU and raises RuntimeError "slow_conv2d_cpu" not implemented for 'Half'

COPY . /app/
WORKDIR /app/

CMD ["python", "main.py"]
