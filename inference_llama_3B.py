import torch, time
from transformers import pipeline
def main():
    model_id = "meta-llama/Llama-3.2-3B"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token="<HuggingFaceToken>"
    )

    start_time = time.time()
    print(pipe("The key to life is"))
    print(f"time taken: {time.time() - start_time}s")

if __name__ == '__main__':
    main()