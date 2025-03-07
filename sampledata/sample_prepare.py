import os
import numpy as np
from tokenizers import Tokenizer

input_files = [
    ('sample_ZuluMax.txt', 'sample_ZuluMax.bin'),
    ('sample_YorubaMax.txt', 'sample_YorubaMax.bin'),
    ('sample_ZuluPrompt.txt', 'sample_ZuluPrompt.bin'),
    ('sample_YorubaPrompt.txt', 'sample_YorubaPrompt.bin'),
    ('sample_ZuluMini.txt', 'sample_ZuluMini.bin'),
    ('sample_YorubaMini.txt', 'sample_YorubaMini.bin')
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
    print(f"{input_file} sample stories has {num_tokens} tokens")

    train_ids = np.array(enc.encode(data).ids, dtype=np.uint16)
    train_ids.tofile(output_file)

    print(f"{input_file} sample stories data saved to {output_file}")
