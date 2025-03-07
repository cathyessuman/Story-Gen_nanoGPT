import os
import pickle
import argparse
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# ---------------------------------------------------------------------------
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate text using a trained GPT model.")
parser.add_argument("--out_dir", type=str, default="out", help="Directory containing the model checkpoint.")
parser.add_argument("--ckpoint", type=str, default="ckpt.pt", help="Checkpoint file to load.")
parser.add_argument("--start", type=str, default="K", help="Initial text or prompt for generation.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model settings
init_from = 'resume'  # Load from checkpoint directory
out_dir = args.out_dir  # Model output directory
ckpt_path = os.path.join(out_dir, args.ckpoint)  # Use specified checkpoint
start = args.start  # Prompt input
num_samples = 1000 # number of samples to draw
max_new_tokens = 512 # number of tokens generated in each sample
temperature = 0.7 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 50 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337

seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Use PyTorch 2.0 to compile the model

# ---------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model checkpoint
if init_from == 'resume':
    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # Requires PyTorch 2.0

# Load tokenizer
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    # decode = lambda l: ''.join([itos[i] for i in l])
    def decode(l):
        text = ''.join([itos[i] for i in l])
        text = text.replace("Ġ", " ")  
        text = text.replace("Ċ", "\n")
        return text.strip()
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


# Encode prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# Run generation
END_TOKEN = "<|endofstory|>"

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

            generated_text = decode(y[0].tolist())

            if END_TOKEN in generated_text:
                generated_text = generated_text.split(END_TOKEN)[0] + END_TOKEN  # Keep only the first complete story
            
            print(generated_text)
            print('---------------')
