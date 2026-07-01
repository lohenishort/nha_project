import torch
import torch.nn as nn
import torch.nn.functional as F

def slot_update_step(k_t, v_t, K_prev, V_prev, alpha_t):
    """
    One recurrent step of the gated long-term memory (unbatched).
    k_t, v_t   : (d,)      current token's key/value
    K_prev     : (m, d)    previous slot keys
    V_prev     : (m, d)    previous slot values
    alpha_t    : (m,)      per-slot gate in (0,1), sigmoid already applied
    returns K_t, V_t : (m, d)
    """
    K_t = alpha_t[:, None] * K_prev + (1 - alpha_t)[:, None] * k_t[None, :]
    V_t = alpha_t[:, None] * V_prev + (1 - alpha_t)[:, None] * v_t[None, :]
    return K_t, V_t

def slot_update_step_batched(k_t, v_t, K_prev, V_prev, alpha_t):
    """
    One recurrent step of the gated long-term memory (batched).
    k_t, v_t   : (batch_size, d)      current token's key/value
    K_prev     : (batch_size, m, d)    previous slot keys
    V_prev     : (batch_size, m, d)    previous slot values
    alpha_t    : (batch_size, m)      per-slot gate in (0,1), sigmoid already applied
    returns K_t, V_t : (batch_size, m, d)
    """
    alpha_t_unsqueezed = alpha_t.unsqueeze(-1)  # (batch_size, m, 1)
    K_t = alpha_t_unsqueezed * K_prev + (1.0 - alpha_t_unsqueezed) * k_t.unsqueeze(1)
    V_t = alpha_t_unsqueezed * V_prev + (1.0 - alpha_t_unsqueezed) * v_t.unsqueeze(1)
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

def nha_layer_recurrent(x_seq, W_q, W_k, W_v, W_alpha, m, window, cos, sin, W_mem=None, ablate_slots=False):
    """
    Recurrent Nearest Neighbor Attention layer (vectorized across batches).
    x_seq      : (batch_size, seq_len, d_model) or (seq_len, d_model)
    """
    is_batched = (x_seq.dim() == 3)
    if not is_batched:
        x_seq = x_seq.unsqueeze(0)  # Convert to (1, seq_len, d_model)

    batch_size, seq_len, d_model = x_seq.shape
    d = W_q.out_features

    cos_unsqueezed = cos.unsqueeze(0)  # (1, seq_len, d)
    sin_unsqueezed = sin.unsqueeze(0)  # (1, seq_len, d)

    q_seq_unrotated = W_q(x_seq)  # (batch_size, seq_len, d)
    q_seq = apply_rope(q_seq_unrotated, cos_unsqueezed, sin_unsqueezed)

    k_seq_unrotated = W_k(x_seq)  # (batch_size, seq_len, d)
    k_seq = apply_rope(k_seq_unrotated, cos_unsqueezed, sin_unsqueezed)

    v_seq = W_v(x_seq)  # (batch_size, seq_len, d)

    K_long = torch.zeros(batch_size, m, d, device=x_seq.device)
    V_long = torch.zeros(batch_size, m, d, device=x_seq.device)
    slots_written = False

    outputs = []
    for t in range(seq_len):
        exit_idx = t - window
        if exit_idx >= 0 and not ablate_slots:
            alpha_logits = W_alpha(x_seq[:, exit_idx])  # (batch_size, m)
            if W_mem is not None:
                mem_cat = torch.cat([K_long, V_long], dim=-1)  # (batch_size, m, 2*d)
                gate_feedback = W_mem(mem_cat).squeeze(-1)  # (batch_size, m)
                alpha_logits = alpha_logits + gate_feedback
            alpha_t = torch.sigmoid(alpha_logits)

            k_exit_raw = k_seq_unrotated[:, exit_idx]  # (batch_size, d)
            v_exit_raw = v_seq[:, exit_idx]  # (batch_size, d)
            K_long, V_long = slot_update_step_batched(k_exit_raw, v_exit_raw, K_long, V_long, alpha_t)
            slots_written = True

        win_start = max(0, t - window + 1)
        win_len = t + 1 - win_start
        K_short = k_seq[:, win_start:t+1]  # (batch_size, win_len, d)
        V_short = v_seq[:, win_start:t+1]  # (batch_size, win_len, d)

        q_t_rotated = q_seq[:, t]  # (batch_size, d)
        q_t_unrotated = q_seq_unrotated[:, t]  # (batch_size, d)

        if slots_written:
            # Exclude RoPE for K_long by using q_t_unrotated, but use q_t_rotated for K_short
            scores_long = torch.bmm(q_t_unrotated.unsqueeze(1), K_long.transpose(1, 2)).squeeze(1) / (d ** 0.5)  # (batch_size, m)
            scores_short = torch.bmm(q_t_rotated.unsqueeze(1), K_short.transpose(1, 2)).squeeze(1) / (d ** 0.5)  # (batch_size, win_len)
            scores = torch.cat([scores_long, scores_short], dim=-1)  # (batch_size, m + win_len)
            V_H = torch.cat([V_long, V_short], dim=1)  # (batch_size, m + win_len, d)
        else:
            scores = torch.bmm(q_t_rotated.unsqueeze(1), K_short.transpose(1, 2)).squeeze(1) / (d ** 0.5)  # (batch_size, win_len)
            V_H = V_short  # (batch_size, win_len, d)

        weights = F.softmax(scores, dim=-1)  # (batch_size, slots + win_len)
        out_t = torch.bmm(weights.unsqueeze(1), V_H).squeeze(1)  # (batch_size, d)
        outputs.append(out_t)

    res = torch.stack(outputs, dim=1)  # (batch_size, seq_len, d)
    if not is_batched:
        res = res.squeeze(0)
    return res

def nha_layer_recurrent_ablate_slots(x_seq, W_q, W_k, W_v, W_alpha, m, window, cos, sin):
    """Ablation utility: long-term slots are never written."""
    return nha_layer_recurrent(
        x_seq, W_q, W_k, W_v, W_alpha, m, window, cos, sin, ablate_slots=True
    )

class NHALayer(nn.Module):
    def __init__(self, d_model, d_head, m, window, ablate_slots=False):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.m = m
        self.window = window
        self.ablate_slots = ablate_slots

        self.W_q = nn.Linear(d_model, d_head)
        self.W_k = nn.Linear(d_model, d_head)
        self.W_v = nn.Linear(d_model, d_head)
        self.W_o = nn.Linear(d_head, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model)
        )

        if not ablate_slots:
            self.W_alpha = nn.Linear(d_model, m)
            # Initialize write gate bias to a high value to avoid vanishing gradients (memory decay)
            nn.init.constant_(self.W_alpha.bias, 3.0)
            self.W_mem = nn.Linear(2 * d_head, 1)
            # Initialize feedback weights to small values
            nn.init.xavier_uniform_(self.W_mem.weight)
            nn.init.zeros_(self.W_mem.bias)
        else:
            self.W_alpha = None
            self.W_mem = None

    def forward(self, x_seq, cos, sin, force_ablate_slots=False):
        """
        x_seq: (batch_size, seq_len, d_model)
        cos, sin: (seq_len, d_head)
        """
        # Pre-LN Attention
        norm_x = self.ln1(x_seq)
        ablate = self.ablate_slots or force_ablate_slots
        h = nha_layer_recurrent(
            norm_x, self.W_q, self.W_k, self.W_v, self.W_alpha,
            self.m, self.window, cos, sin,
            W_mem=self.W_mem, ablate_slots=ablate
        )
        x_seq = x_seq + self.W_o(h)
        
        # Pre-LN MLP
        x_seq = x_seq + self.mlp(self.ln2(x_seq))
        return x_seq

class TinyNHAModel(nn.Module):
    def __init__(self, vocab_total, d_model, d_head, m, window, max_seq_len, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_total, d_model)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            ablate = (i < num_layers - 1)
            self.layers.append(NHALayer(d_model, d_head, m, window, ablate_slots=ablate))
            
        self.out_head = nn.Linear(d_model, vocab_total)
        self.cos, self.sin = build_rope_cache(max_seq_len, d_head)

    def forward(self, x_ids, force_ablate_slots=False):
        batch_size, seq_len = x_ids.shape
        h = self.embed(x_ids)  # (batch_size, seq_len, d_model)
        
        cos_slice = self.cos[:seq_len].to(x_ids.device)
        sin_slice = self.sin[:seq_len].to(x_ids.device)
        
        for layer in self.layers:
            h = layer(h, cos_slice, sin_slice, force_ablate_slots=force_ablate_slots)
            
        return self.out_head(h)

    @property
    def W_q(self):
        return self.layers[-1].W_q

    @property
    def W_k(self):
        return self.layers[-1].W_k

    @property
    def W_v(self):
        return self.layers[-1].W_v

    @property
    def W_alpha(self):
        return self.layers[-1].W_alpha
