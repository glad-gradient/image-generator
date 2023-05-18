FROM python:3.10.11

# Install any necessary dependencies
# RUN apt-get install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update
# RUN apt-get install -y python3.8

# RUN apt update && apt upgrade -y
# RUN apt install python3 -y

RUN apt-get update && apt-get install -y \
wget

RUN wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image.py -P app/

RUN pip install --upgrade pip
# ENV LD_LIBRARY_PATH="/usr/lib/wsl/lib/"
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/"
RUN pip3 uninstall torch torchvision -y
RUN pip uninstall bitsandbytes -y

COPY requirements.txt /app/
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r /app/requirements.txt

# RUN accelerate config default # Sets the device to CPU and raises RuntimeError "slow_conv2d_cpu" not implemented for 'Half'

COPY . /app/
WORKDIR /app/

CMD ["python", "main.py"]
