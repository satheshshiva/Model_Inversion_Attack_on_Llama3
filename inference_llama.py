import torch
from transformers import pipeline
import os
def main():
    model_id = "meta-llama/Llama-3.2-1B"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token="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"
    )

    print(pipe("The key to life is"))

if __name__ == '__main__':
    main()