import torch
import torch.nn as nn
import torch.nn.functional as F

def slot_update_step(k_t, v_t, K_prev, V_prev, alpha_t):
    """
    One recurrent step of the gated long-term memory.
    k_t, v_t   : (d,)      current token's key/value
    K_prev     : (m, d)    previous slot keys
    V_prev     : (m, d)    previous slot values
    alpha_t    : (m,)      per-slot gate in (0,1), sigmoid already applied
    returns K_t, V_t : (m, d)
    """
    K_t = alpha_t[:, None] * K_prev + (1 - alpha_t)[:, None] * k_t[None, :]
    V_t = alpha_t[:, None] * V_prev + (1 - alpha_t)[:, None] * v_t[None, :]
    return K_t, V_t

def build_rope_cache(seq_len, d, base=10000.0):
    assert d % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)          # (seq_len, d/2)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (seq_len, d)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin

def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

def nha_layer_recurrent(x_seq, W_q, W_k, W_v, W_alpha, m, window, cos, sin):
    seq_len, d_model = x_seq.shape
    d = W_q.out_features

    q_seq = apply_rope(W_q(x_seq), cos, sin)
    k_seq = apply_rope(W_k(x_seq), cos, sin)
    v_seq = W_v(x_seq)

    K_long = torch.zeros(m, d, device=x_seq.device)
    V_long = torch.zeros(m, d, device=x_seq.device)
    slots_written = False

    outputs = []
    for t in range(seq_len):
        exit_idx = t - window
        if exit_idx >= 0:
            alpha_t = torch.sigmoid(W_alpha(x_seq[exit_idx]))
            k_exit_raw = W_k(x_seq[exit_idx])
            v_exit_raw = W_v(x_seq[exit_idx])
            K_long, V_long = slot_update_step(k_exit_raw, v_exit_raw, K_long, V_long, alpha_t)
            slots_written = True

        win_start = max(0, t - window + 1)
        K_short = k_seq[win_start:t+1]
        V_short = v_seq[win_start:t+1]

        if slots_written:
            K_H = torch.cat([K_long, K_short], dim=0)
            V_H = torch.cat([V_long, V_short], dim=0)
        else:
            K_H, V_H = K_short, V_short

        q_t = q_seq[t]
        scores = (q_t @ K_H.T) / (d ** 0.5)
        weights = F.softmax(scores, dim=-1)
        outputs.append(weights @ V_H)

    return torch.stack(outputs, dim=0)

def nha_layer_recurrent_ablate_slots(x_seq, W_q, W_k, W_v, W_alpha, m, window, cos, sin):
    """Ablation utility: long-term slots are never written."""
    seq_len, d_model = x_seq.shape
    d = W_q.out_features
    q_seq = apply_rope(W_q(x_seq), cos, sin)
    k_seq = apply_rope(W_k(x_seq), cos, sin)
    v_seq = W_v(x_seq)
    
    outputs = []
    for t in range(seq_len):
        win_start = max(0, t - window + 1)
        K_short = k_seq[win_start:t+1]
        V_short = v_seq[win_start:t+1]
        q_t = q_seq[t]
        scores = (q_t @ K_short.T) / (d ** 0.5)
        weights = F.softmax(scores, dim=-1)
        outputs.append(weights @ V_short)
    return torch.stack(outputs, dim=0)

class TinyNHAModel(nn.Module):
    def __init__(self, vocab_total, d_model, d_head, m, window, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_total, d_model)
        self.W_q = nn.Linear(d_model, d_head)
        self.W_k = nn.Linear(d_model, d_head)
        self.W_v = nn.Linear(d_model, d_head)
        self.W_alpha = nn.Linear(d_model, m)
        self.out_head = nn.Linear(d_head, vocab_total)
        self.m = m
        self.window = window
        self.cos, self.sin = build_rope_cache(max_seq_len, d_head)

    def forward(self, x_ids):
        batch_size, seq_len = x_ids.shape
        logits_batch = []
        for b in range(batch_size):
            x_seq = self.embed(x_ids[b])
            h = nha_layer_recurrent(
                x_seq, self.W_q, self.W_k, self.W_v, self.W_alpha,
                self.m, self.window, 
                self.cos[:seq_len].to(x_ids.device), 
                self.sin[:seq_len].to(x_ids.device)
            )
            logits_batch.append(self.out_head(h))
        return torch.stack(logits_batch, dim=0)
