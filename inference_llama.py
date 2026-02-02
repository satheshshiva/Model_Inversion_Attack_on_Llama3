import torch, time
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

    print("Started generation")
    start_time = time.time()
    text_input = 'account number:'

    print(pipe(text_input,
               max_new_tokens=50,
               num_return_sequences=1,
               top_p=1,
               top_k=40
               ))
    print(f"time taken: {time.time() - start_time}s")

if __name__ == '__main__':
    main()