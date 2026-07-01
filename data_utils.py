import torch

def make_recall_batch(batch_size, seq_len, vocab_size, key_val_gap):
    """
    Sequence layout per example:
      [KEY_TOK, key_id, VAL_TOK, val_id, ...filler..., QUERY_TOK, key_id, ANSWER_POS]
    Model must predict val_id at ANSWER_POS.
    Token ids: 0=KEY_TOK, 1=VAL_TOK, 2=QUERY_TOK, 3=filler, 4..(4+vocab_size)=content ids
    """
    assert key_val_gap + 6 <= seq_len
    x = torch.full((batch_size, seq_len), 3, dtype=torch.long)  # filler everywhere
    target_pos = torch.zeros(batch_size, dtype=torch.long)
    target_val = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        key_id = torch.randint(4, 4 + vocab_size, (1,)).item()
        val_id = torch.randint(4, 4 + vocab_size, (1,)).item()

        x[b, 0] = 0          # KEY_TOK
        x[b, 1] = key_id
        x[b, 2] = 1          # VAL_TOK
        x[b, 3] = val_id

        query_pos = 4 + key_val_gap
        x[b, query_pos] = 2       # QUERY_TOK
        x[b, query_pos + 1] = key_id
        answer_pos = query_pos + 2
        
        target_pos[b] = answer_pos - 1
        target_val[b] = val_id

    return x, target_pos, target_val
