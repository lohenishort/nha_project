#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from data_utils import make_recall_batch
from nha_core import TinyNHAModel

def main():
    parser = argparse.ArgumentParser(description="Train NHA model on synthetic or real language modeling data.")
    parser.add_argument("--data-mode", type=str, choices=["synthetic", "real"], default="synthetic",
                        help="Data mode: synthetic KV-recall or real language modeling.")
    parser.add_argument("--dataset", type=str, default="c4",
                        help="Hugging Face dataset name for real language modeling.")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for real language modeling.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for real language modeling.")
    parser.add_argument("--steps", type=int, default=301,
                        help="Number of training steps.")
    args = parser.parse_args()

    if args.data_mode == "synthetic":
        # Setup seed
        torch.manual_seed(0)

        # Hyperparameters
        vocab_size = 20
        vocab_total = 4 + vocab_size
        d_model, d_head, m = 32, 16, 8
        window = 8
        key_val_gap = 24
        seq_len = key_val_gap + 6

        print("--- Training Main Model (Window = 8, 2 Layers) ---")
        model = TinyNHAModel(vocab_total, d_model, d_head, m, window, max_seq_len=seq_len, num_layers=2)
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

        print("\n--- Training Window=0 Model (Baseline, 2 Layers) ---")
        model_w0 = TinyNHAModel(vocab_total, d_model, d_head, m, window=0, max_seq_len=seq_len, num_layers=2)
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
            logits = model(x, force_ablate_slots=True)
            batch_idx = torch.arange(x.shape[0])
            pred_logits = logits[batch_idx, target_pos]
            acc = (pred_logits.argmax(dim=-1) == target_val).float().mean().item()
            print(f"Accuracy with slots forcibly disabled: {acc:.4f}")

    elif args.data_mode == "real":
        from data_pipeline import LanguageDataLoader
        
        # Setup seed
        torch.manual_seed(0)
        
        print(f"--- Training NHA Model on Real Language Data ({args.dataset}) ---")
        
        loader = LanguageDataLoader(
            dataset_name=args.dataset,
            tokenizer_name="gpt2",
            batch_size=args.batch_size,
            seq_len=args.seq_len
        )
        
        vocab_total = loader.enc.n_vocab
        d_model, d_head, m = 32, 16, 8
        window = 8
        
        model = TinyNHAModel(vocab_total, d_model, d_head, m, window, max_seq_len=args.seq_len, num_layers=2)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        
        iter_loader = iter(loader)
        
        for step in range(args.steps):
            try:
                input_ids, target_ids, attention_mask = next(iter_loader)
            except StopIteration:
                iter_loader = iter(loader)
                input_ids, target_ids, attention_mask = next(iter_loader)
                
            logits = model(input_ids)
            
            # Compute loss over full sequence with attention mask
            loss_flat = F.cross_entropy(logits.view(-1, vocab_total), target_ids.view(-1), reduction='none')
            loss = (loss_flat * attention_mask.view(-1)).sum() / attention_mask.sum().clamp(min=1.0)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Print status periodically
            print_freq = max(1, args.steps // 10)
            if step % print_freq == 0 or step == args.steps - 1:
                # Compute accuracy and perplexity
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    correct = (pred == target_ids).float() * attention_mask
                    acc = correct.sum() / attention_mask.sum().clamp(min=1.0)
                    perplexity = torch.exp(loss)
                print(f"step {step:3d}  loss {loss.item():.4f}  ppl {perplexity.item():.2f}  acc {acc.item():.2f}")

if __name__ == "__main__":
    main()
