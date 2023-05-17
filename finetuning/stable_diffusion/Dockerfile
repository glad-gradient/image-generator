FROM python:3.8

# Install any necessary dependencies
RUN apt-get update && apt-get install -y \
wget \
nvidia-container-toolkit

RUN wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image.py

RUN pip install --upgrade pip
COPY finetuning/stable_diffusion/requirements.txt /app/
RUN pip install -r /app/requirements.txt

RUN accelerate config default

COPY ./finetuning/stable_diffusion/ /app/
WORKDIR /app/

CMD ["python", "main.py"]