out_dir = 'out-yoruba10k'
# eval stuff
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AIMS-paper'
wandb_run_name='yoruba-mini'

dataset = 'yoruba_10k'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 1


n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
