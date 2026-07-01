import torch
import torch.nn.functional as F
from data_utils import make_recall_batch
from nha_core import TinyNHAModel, nha_layer_recurrent_ablate_slots

def main():
    # Setup seed
    torch.manual_seed(0)

    # Hyperparameters
    vocab_size = 20
    vocab_total = 4 + vocab_size
    d_model, d_head, m = 32, 16, 8
    window = 8
    key_val_gap = 24
    seq_len = key_val_gap + 6

    print("--- Training Main Model (Window = 8) ---")
    model = TinyNHAModel(vocab_total, d_model, d_head, m, window, max_seq_len=seq_len)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    for step in range(301):
        x, target_pos, target_val = make_recall_batch(
            batch_size=8, seq_len=seq_len, vocab_size=vocab_size, key_val_gap=key_val_gap
        )
        logits = model(x)

        batch_idx = torch.arange(x.shape[0])
        pred_logits = logits[batch_idx, target_pos]

        loss = F.cross_entropy(pred_logits, target_val)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 30 == 0:
            acc = (pred_logits.argmax(dim=-1) == target_val).float().mean().item()
            print(f"step {step:3d}  loss {loss.item():.4f}  acc {acc:.2f}")

    print("\n--- Training Window=0 Model (Baseline) ---")
    model_w0 = TinyNHAModel(vocab_total, d_model, d_head, m, window=0, max_seq_len=seq_len)
    opt_w0 = torch.optim.Adam(model_w0.parameters(), lr=3e-3)
    
    for step in range(301):
        x, target_pos, target_val = make_recall_batch(
            batch_size=8, seq_len=seq_len, vocab_size=vocab_size, key_val_gap=key_val_gap
        )
        logits = model_w0(x)
        batch_idx = torch.arange(x.shape[0])
        pred_logits = logits[batch_idx, target_pos]
        loss = F.cross_entropy(pred_logits, target_val)
        
        opt_w0.zero_grad()
        loss.backward()
        opt_w0.step()
        
        if step % 30 == 0:
            acc = (pred_logits.argmax(dim=-1) == target_val).float().mean().item()
            print(f"step {step:3d}  loss {loss.item():.4f}  acc {acc:.2f}")

    print("\n--- Ablation Test: Forcing slots disabled on Main Model ---")
    model.eval()
    with torch.no_grad():
        x, target_pos, target_val = make_recall_batch(
            batch_size=32, seq_len=seq_len, vocab_size=vocab_size, key_val_gap=key_val_gap
        )
        logits_batch = []
        for b in range(x.shape[0]):
            x_seq = model.embed(x[b])
            h = nha_layer_recurrent_ablate_slots(
                x_seq, model.W_q, model.W_k, model.W_v, model.W_alpha, 
                m, window, model.cos[:seq_len], model.sin[:seq_len]
            )
            logits_batch.append(model.out_head(h))
            
        logits = torch.stack(logits_batch, dim=0)
        batch_idx = torch.arange(x.shape[0])
        pred_logits = logits[batch_idx, target_pos]
        acc = (pred_logits.argmax(dim=-1) == target_val).float().mean().item()
        print(f"Accuracy with slots forcibly disabled: {acc:.4f}")

if __name__ == "__main__":
    main()
