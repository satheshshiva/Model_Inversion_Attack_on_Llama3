from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GraniteForCausalLM, GraniteConfig, \
    AutoConfig, pipeline, GPT2LMHeadModel
import torch

# model_name = "gpt2-xl"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )
#
# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              device_map=device,
#                                              torch_dtype=torch.bfloat16
#                                              )
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.eval()
#
# # change input text as desired
# chat = [
#     # { "role": "system", "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 25, 2024.\nYou are SecurityGPT, developed by Security Engineering Team at American Express. You are a helpful AI assistant at American Express to help American Express employees with queries related to security tools developed within American Express." },
#     { "role": "user", "content": "who are you?" },
# ]
# chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# # tokenize the text
# input_tokens = tokenizer(chat, return_tensors="pt").to(device)
# # generate output tokens
# output = model.generate(**input_tokens,
#                         max_new_tokens=500,
#                         do_sample=True,
#                         top_p=0.95,
#                         temperature=0.7)
# # decode output tokens into text
# output = tokenizer.batch_decode(output)
# # print output
# print(output)
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
def main():
    model_name = "gpt2-xl"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    text = 'Steam original app owner: '
    encoded_input = tokenizer(text, return_tensors='pt')
    print(f"started generation..")
    start_time = time.time()
    output = model.generate(**encoded_input,
                            max_new_tokens=256,
                            do_sample=True,
                            top_p=1,
                            top_k=40)
    print(f"time taken: {time.time() - start_time}s")
    op = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(op)

if __name__ == '__main__':
    main()