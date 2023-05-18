import json
import subprocess
import logging
from multiprocessing import cpu_count

import torch

logging.basicConfig(level=logging.INFO)


def main():
    # print(torch.cuda.is_available())
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # print('*******************************************')

    with open("configs.json") as f:
        configs = json.load(f)

    max_train_steps = configs["MAX_TRAIN_STEPS"]
    learning_rate = configs["LEARNING_RATE"]
    batch_size = configs["BATCH_SIZE"]
    gradient_accumulation_steps = configs["GRADIENT_ACCUMULATION_STEPS"]
    # n_cpu = cpu_count()
    n_cpu = 1
    validation_prompt = configs["VALIDATION_PROMPT"]
    checkpointing_steps = configs["CHECKPOINTING_STEPS"]
    validation_epochs = configs["VALIDATION_EPOCHS"]
    MODEL_NAME = "CompVis/stable-diffusion-v1-4"
    TRAIN_DIR = configs["TRAIN_DIR"]
    OUTPUT_DIR = configs["OUTPUT_DIR"]

    with subprocess.Popen(
        [
            "accelerate", "launch", "train_text_to_image.py",
            f"--pretrained_model_name_or_path={MODEL_NAME}",
            f"--train_data_dir={TRAIN_DIR}",
            "--use_ema",
            "--resolution=512", "--center_crop", "--random_flip",
            f"--train_batch_size={batch_size}",
            f"--gradient_accumulation_steps={gradient_accumulation_steps}",  # default=1
            "--gradient_checkpointing",
            "--mixed_precision=fp16",
            f"--max_train_steps={max_train_steps}",
            f"--learning_rate={learning_rate}",
            "--max_grad_norm=1",
            "--lr_scheduler=constant", "--lr_warmup_steps=0",
            f"--output_dir={OUTPUT_DIR}",
            # f"--resume_from_checkpoint={}",
            # "--enable_xformers_memory_efficient_attention",
            f"--dataloader_num_workers={n_cpu}",
            f"--validation_prompts={validation_prompt}", # str
            f"--checkpointing_steps={checkpointing_steps}", # default=500,
            f"--validation_epochs={validation_epochs}", #  default=5,
        ],
        stdout=subprocess.PIPE,
        universal_newlines=True
    ) as process:
        while True:
            output = process.stdout.readline()
            logging.info(output.strip())
            # Do something else
            return_code = process.poll()
            if return_code is not None:
                logging.info(f"RETURN CODE: {return_code}")
                # Process has finished, read rest of the output
                for output in process.stdout.readlines():
                    logging.info(output.strip())
                break


if __name__ == "__main__":
    main()
