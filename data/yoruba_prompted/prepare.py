from tokenizers import Tokenizer
import numpy as np

input_file_path = '/content/nanoGPT/data/yoruba_prompted/yoruba_gen_stories.txt'

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# enc = tiktoken.get_encoding("gpt2")
chunksize = 1024

enc = Tokenizer.from_file("yoruba_tokenizer.json")
train_ids = []
val_ids = []

for i in range(0, len(train_data), chunksize):
  chunk = train_data[i:i+chunksize]
  train_ids.extend(enc.encode(chunk).ids)


for i in range(0, len(val_data), chunksize):
  chunk = val_data[i:i+chunksize]
  val_ids.extend(enc.encode(chunk).ids)

# train_ids = np.array(train_ids.ids, dtype=np.uint16)
# val_ids = np.array(val_ids.ids, dtype=np.uint16)

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_output_path = 'train.bin'
val_output_path = 'val.bin'

train_ids.tofile(train_output_path)
val_ids.tofile(val_output_path)

print(f"Training data saved to {train_output_path}")
print(f"Validation data saved to {val_output_path}")

train_data = np.fromfile('train.bin', dtype=np.uint16)
val_data = np.fromfile('val.bin', dtype=np.uint16)

print("First 10 tokens in training data:", train_data[:10])
print("First 10 tokens in validation data:", val_data[:10])
