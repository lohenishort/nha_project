# Nearest Neighbor Attention (NHA) Long-Term Memory Model

This repository implements a recurrent, gated long-term memory model called `TinyNHAModel` using **Nearest Neighbor Attention (NHA)**. It is tested on a synthetic associative Key-Value (KV) recall task.

---

## 1. Project Architecture

The model consists of the following components:
*   `nha_core.py`: Implementation of `NHALayer` and `TinyNHAModel`. It encapsulates the recurrent Nearest Neighbor Attention layer, Rotary Position Embeddings (RoPE), memory slot gating, and vectorized updates.
*   `data_utils.py`: Synthetic KV recall task generator. The sequence layout is:
    ```
    [KEY_TOK, key_id, VAL_TOK, val_id, ...filler..., QUERY_TOK, key_id, ANSWER_POS]
    ```
    The model must predict `val_id` at `ANSWER_POS`.
*   `train.py`: Training script for both the Main Model (`window=8`) and the Baseline Model (`window=0`), plus an ablation test that evaluates accuracy with slots disabled.
*   `test_nha.py`: Unit test suite testing core utility functions, vectorized recurrence, and model shapes.

---

## 2. Solved Bottlenecks & Mathematical Fixes

### 🚨 A. Mathematically Impossible Association (Single-Layer Limitation)
*   **The Issue:** A single-layer model writes embedding representations to long-term memory slots directly. At step `exit_idx = 3` (where `val_id` resides), the token input has zero awareness of `key_id` at index 1. Thus, the written slot has no mathematical binding between key and value.
*   **The Solution:** Stacked the model into 2 layers.
    *   **Layer 1:** Uses local window attention (`ablate_slots=True`) to mix `key_id` and `val_id` within the local window.
    *   **Layer 2:** Reads the mixed representation at position 3 and writes it to long-term slots, establishing the association.

### 🚨 B. Mismatched Rotary Position Embeddings (RoPE)
*   **The Issue:** Rotary Position Embeddings (RoPE) were applied to short-term keys `K_short` and queries `q_t`, but long-term keys `K_long` were saved raw without RoPE. Taking the dot product `q_t @ K_long.T` modulated scores periodically based on position $t$, preventing stable retrieval.
*   **The Solution:** Kept `K_long` unrotated and computed separate unrotated queries specifically for long-term memory retrieval:
    *   `scores_long = q_unrotated @ K_long.T` (position-agnostic retrieval)
    *   `scores_short = q_rotated @ K_short.T` (position-aware relative short-term retrieval)

### 🚨 C. Feed-Forward Gating Bottleneck (Vanishing Gradients)
*   **The Issue:** The write gate $\alpha_t$ was purely feed-forward ($\alpha_t = \sigma(W_{\alpha} x_t)$). Since `alpha_t` defaults to $\approx 0.5$ at initialization, the memory slots decay by $50\%$ at every step. Over a 24-step key-value gap, the memory decays by $0.5^{24} \approx 0$, causing vanishing gradients.
*   **The Solution:**
    1.  Conditioned the write gate computation on both the input token and the previous memory state.
    2.  Initialized the gate bias to `3.0` (setting gate retention to $\approx 95\%$ by default), preventing early decay and enabling robust gradient flow.

---

## 3. Vectorization & Parallelization

*   **The Issue:** `TinyNHAModel.forward` previously looped sequentially over batch items using a Python `for` loop, causing severe GPU utility bottlenecks.
*   **The Solution:** Vectorized `nha_layer_recurrent` and `slot_update_step_batched` to accept 3D batched inputs of shape `(batch_size, seq_len, d_model)` and perform slot updates across all batch elements in parallel. Backward compatibility is maintained for unbatched 2D inputs.

---

## 4. Usage

### Install Dependencies
To setup the environment and install dependencies:
```bash
uv pip install -e .
```

### Run Unit Tests
To run the unittest suite:
```bash
uv run python3 -m unittest test_nha.py
```

### Train the Models
To train the main model and baseline:
```bash
uv run train.py
```
