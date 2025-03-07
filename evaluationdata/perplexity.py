import os
import numpy as np
import torch
from torch.nn import functional as F
from contextlib import nullcontext
from model import GPTConfig, GPT

# Configuration
CHECKPOINT_NAME = "ckpt_iter_5000.pt"  # Modify if needed
BLOCK_SIZE = 256  # Model's expected input size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CTX = nullcontext() if DEVICE == 'cpu' else torch.amp.autocast(device_type=DEVICE, dtype=torch.float16)

# Define model directories (Max models) and corresponding evaluation datasets
MODEL_EVALS = {
    "ZuluMax": ("/content/nanoGPT/out-zulu", [
        "/content/nanoGPT/evaluationdata/eval_ZuluMax.bin",
        "/content/nanoGPT/evaluationdata/eval_ZuluPrompt.bin",
        "/content/nanoGPT/evaluationdata/eval_ZuluMini.bin"
    ]),
    "YorubaMax": ("/content/nanoGPT/out-yoruba", [
        "/content/nanoGPT/evaluationdata/eval_YorubaMax.bin",
        "/content/nanoGPT/evaluationdata/eval_YorubaPrompt.bin",
        "/content/nanoGPT/evaluationdata/eval_YorubaMini.bin"
    ])
}


def load_eval_data(eval_path):
    """ Load tokenized evaluation data from .bin files """
    if not os.path.exists(eval_path):
        print(f"Sample file not found: {eval_path}, skipping...")
        return None
    return np.memmap(eval_path, dtype=np.uint16, mode='r')


def calculate_perplexity(model, eval_data):
    """ Compute perplexity over the **entire** evaluation sample """
    if eval_data is None:
        print("Skipping perplexity computation due to missing sample data.")
        return float('inf')

    model.eval()
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        # Process the entire dataset in chunks of BLOCK_SIZE
        for i in range(0, len(eval_data) - BLOCK_SIZE, BLOCK_SIZE):
            x = torch.tensor(eval_data[i:i+BLOCK_SIZE], dtype=torch.long, device=DEVICE)[None, ...]
            y = torch.tensor(eval_data[i+1:i+1+BLOCK_SIZE], dtype=torch.long, device=DEVICE)[None, ...]

            with CTX:
                logits, loss = model(x, y)

            log_probs = F.log_softmax(logits, dim=-1)

            # Flatten for loss calculation
            y = y.view(-1)
            log_probs = log_probs.view(-1, log_probs.size(-1))

            loss = F.nll_loss(log_probs, y, reduction='sum')

            total_loss += loss.item()
            total_count += y.numel()

    avg_loss = total_loss / total_count if total_count > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    return perplexity


def load_model(model_dir):
    """ Load a trained model from checkpoint """
    ckpt_path = os.path.join(model_dir, CHECKPOINT_NAME)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {CHECKPOINT_NAME} not found in {model_dir}, skipping...")
        return None

    print(f"Loading model from {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # Load state dictionary
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(DEVICE)

    return model


def main():
    for model_name, (model_dir, eval_paths) in MODEL_EVALS.items():
        model = load_model(model_dir)
        if model is None:
            continue  # Skip if model loading fails

        for eval_path in eval_paths:
            eval_data = load_eval_data(eval_path)
            perplexity = calculate_perplexity(model, eval_data)
            eval_name = os.path.basename(eval_path).replace('.bin', '')  # Get clean evaluation name
            print(f"Perplexity for {eval_name} using {model_name}: {perplexity:.4f}")


if __name__ == "__main__":
    main()
