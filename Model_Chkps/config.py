from dataclasses import dataclass

@dataclass
class ModelArgs:
    in_ = len(vocab)
    batch = 128
    seq_len = 128
    dim = 256
    q_heads = 8
    kv_heads = 2
    hdim = dim // q_heads
    blocks = 4
    dropout = 0.2

@dataclass
class TrainingArgs:
    epochs = 1e2
    lr = 1e-3
    scheduler_gamma = 0.9999
    use_mixed_precision = True
    model_save_dir = '' 
    model_load_dir = model_save_dir
    loss_dir = ''
