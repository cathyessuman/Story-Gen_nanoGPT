import os
import numpy as np
from tokenizers import Tokenizer

input_files = [
    ('eval_ZuluMax.txt', 'eval_ZuluMax.bin'),
    ('eval_YorubaMax.txt', 'eval_YorubaMax.bin'),
    ('eval_ZuluPrompt.txt', 'eval_ZuluPrompt.bin'),
    ('eval_YorubaPrompt.txt', 'eval_YorubaPrompt.bin'),
    ('eval_ZuluMini.txt', 'eval_ZuluMini.bin'),
    ('eval_YorubaMini.txt', 'eval_YorubaMini.bin')
]

chunksize = 1024

for input_file, output_file in input_files:
    if "zulu" in input_file.lower():
        tokenizer_file = 'zulu_tokenizer.json'
    elif "yoruba" in input_file.lower():
        tokenizer_file = 'yoruba_tokenizer.json'
    else:
        raise ValueError(f"Unknown language in file: {input_file}. Please specify a correct tokenizer.")

    enc = Tokenizer.from_file(tokenizer_file)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    num_tokens = len(enc.encode(data).ids)
    print(f"{input_file} evaluation stories has {num_tokens} tokens")

    train_ids = np.array(enc.encode(data).ids, dtype=np.uint16)
    train_ids.tofile(output_file)

    print(f"{input_file} evaluation stories data saved to {output_file}")