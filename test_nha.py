import unittest
import torch
import torch.nn as nn
from nha_core import (
    slot_update_step_batched,
    build_rope_cache,
    apply_rope,
    nha_layer_recurrent,
    TinyNHAModel
)

class TestNHA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.seq_len = 10
        self.d_model = 32
        self.d_head = 16
        self.m = 4
        self.window = 4

    def test_build_rope_cache(self):
        cos, sin = build_rope_cache(self.seq_len, self.d_head)
        self.assertEqual(cos.shape, (self.seq_len, self.d_head))
        self.assertEqual(sin.shape, (self.seq_len, self.d_head))

    def test_apply_rope(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_head)
        cos, sin = build_rope_cache(self.seq_len, self.d_head)
        
        # Broadcast cos/sin over batch dim
        x_rot = apply_rope(x, cos.unsqueeze(0), sin.unsqueeze(0))
        self.assertEqual(x_rot.shape, x.shape)

    def test_slot_update_step_batched(self):
        k_t = torch.randn(self.batch_size, self.d_head)
        v_t = torch.randn(self.batch_size, self.d_head)
        K_prev = torch.randn(self.batch_size, self.m, self.d_head)
        V_prev = torch.randn(self.batch_size, self.m, self.d_head)
        alpha_t = torch.sigmoid(torch.randn(self.batch_size, self.m))

        K_t, V_t = slot_update_step_batched(k_t, v_t, K_prev, V_prev, alpha_t)
        self.assertEqual(K_t.shape, (self.batch_size, self.m, self.d_head))
        self.assertEqual(V_t.shape, (self.batch_size, self.m, self.d_head))

    def test_nha_layer_recurrent_shapes(self):
        # Batched input
        x_seq_batched = torch.randn(self.batch_size, self.seq_len, self.d_model)
        W_q = nn.Linear(self.d_model, self.d_head)
        W_k = nn.Linear(self.d_model, self.d_head)
        W_v = nn.Linear(self.d_model, self.d_head)
        W_alpha = nn.Linear(self.d_model, self.m)
        W_mem = nn.Linear(2 * self.d_head, 1)

        cos, sin = build_rope_cache(self.seq_len, self.d_head)

        res_batched = nha_layer_recurrent(
            x_seq_batched, W_q, W_k, W_v, W_alpha,
            self.m, self.window, cos, sin, W_mem=W_mem
        )
        self.assertEqual(res_batched.shape, (self.batch_size, self.seq_len, self.d_head))

        # Unbatched input (backward compatibility test)
        x_seq_unbatched = torch.randn(self.seq_len, self.d_model)
        res_unbatched = nha_layer_recurrent(
            x_seq_unbatched, W_q, W_k, W_v, W_alpha,
            self.m, self.window, cos, sin, W_mem=W_mem
        )
        self.assertEqual(res_unbatched.shape, (self.seq_len, self.d_head))

    def test_tiny_nha_model(self):
        vocab_total = 24
        model = TinyNHAModel(vocab_total, self.d_model, self.d_head, self.m, self.window, self.seq_len, num_layers=2)
        
        x = torch.randint(0, vocab_total, (self.batch_size, self.seq_len))
        
        # Test standard forward
        logits = model(x)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, vocab_total))

        # Test forward with forced ablation
        logits_ablate = model(x, force_ablate_slots=True)
        self.assertEqual(logits_ablate.shape, (self.batch_size, self.seq_len, vocab_total))

if __name__ == "__main__":
    unittest.main()
