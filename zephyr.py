from fastapi import FastAPI, Request
from time import time
import torch
import uvicorn
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.bfloat16,
    device=torch.device("mps"),
)

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]

app = FastAPI()


@app.get("/")
async def root(request: Request):
    return {"message": infer()}


def infer():
    now = time()
    print("Infering", now)

    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("Done tokenizing", time() - now)
    outputs = pipe(
        prompt,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    print("Infered", time() - now)
    print()
    return outputs[0]["generated_text"]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
