# nanoGPT `model.py` Explained — Block by Block

A walkthrough of nanoGPT's `model.py` — the whole GPT in ~330 lines. Explained in the order data actually flows, building up from the small pieces. Numbers use the default config to stay concrete: **embedding dim `n_embd=768`, `n_head=12` heads, `block_size=1024` (max sequence length), `n_layer=12` blocks, vocab ~50k.**

## The big picture first

A GPT is a next-token predictor. You feed in a sequence of token IDs, and it outputs, for each position, a probability distribution over "what token comes next." The whole architecture exists to let each token's representation gather information from earlier tokens (that's **attention**) and then do per-token nonlinear processing (that's the **MLP**). You stack that pair 12 times.

Every token is represented as a **vector of 768 numbers** throughout the network. Think of that 768-dim vector as the token's "working memory" — it starts as just "I am the word `cat`" and progressively accumulates context like "I am `cat`, the subject of this sentence, after the word `the`."

---

## 1. LayerNorm (lines 18–27)

```python
return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

A normalization step. For each token's 768-vector, it rescales the numbers so they have mean 0 and variance 1, then applies a learnable scale (`weight`) and shift (`bias`).

**Why it exists:** deep networks become unstable to train when the magnitudes of activations drift across layers. LayerNorm keeps every layer's inputs in a predictable range. nanoGPT's only customization is making `bias` optional (the GPT-2 original always had it; dropping it is slightly faster and marginally better).

You'll see it used **before** attention and **before** the MLP — this is the "pre-norm" design, which I'll come back to in the Block.

---

## 2. CausalSelfAttention (lines 29–76) — the heart of it

This is the only place where tokens **talk to each other**. Everything else processes each token independently. Let me build the intuition before the code.

**The intuition.** Each token produces three vectors:
- A **query** ("what am I looking for?")
- A **key** ("what do I offer to others?")
- A **value** ("what information do I pass along if someone attends to me?")

A token attends to earlier tokens by comparing its *query* against their *keys*. High match → high attention weight. Then it builds its new representation as a weighted sum of those tokens' *values*. So the word `it` can look back, find that `the dog` is the relevant noun, and pull that meaning into itself.

**"Causal"** means a token can only look **leftward** (at itself and earlier tokens), never at the future. That's mandatory for a next-token predictor — peeking ahead would be cheating.

### The code

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
```
One linear layer produces query, key, AND value at once (3 × 768 = 2304 outputs), then splits them. Doing it as one matrix multiply is just an efficiency trick versus three separate ones.

```python
q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
```
Here `B`=batch, `T`=sequence length, `C`=768. This is **multi-head** attention: instead of one attention operation over 768-dim vectors, split into **12 heads** of 64 dims each (768/12). Each head does attention independently.

**Why multiple heads?** One head can only learn one "kind" of relationship per position. With 12, one head might track subject-verb agreement, another might track which adjective modifies which noun, etc. They run in parallel and get concatenated back together.

The `.view(...).transpose(...)` reshapes `(B, T, 768)` into `(B, 12, T, 64)` so the head dimension sits next to the batch dimension — PyTorch then treats each (batch, head) pair as an independent attention problem.

```python
if self.flash:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, ..., is_causal=True)
else:
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v
```

Both branches compute the **same math**; the `flash` branch is just a fused, memory-efficient GPU kernel (PyTorch ≥ 2.0). The `else` branch spells out what's happening, so read that:

1. **`q @ k.transpose`** — every token's query dotted with every token's key. Result is a `T×T` grid of "how much should token i attend to token j" scores.
2. **`/ sqrt(64)`** — scaling so the dot products don't grow huge as head size increases (large values make softmax too "peaky" and gradients vanish).
3. **`masked_fill(... == 0, -inf)`** — this is **causality**. `self.bias` is a lower-triangular matrix of 1s (built on line 49 via `torch.tril`). Wherever it's 0 (the future positions), set the score to `-inf` so that...
4. **`softmax`** — ...those `-inf` scores become exactly 0 weight. Softmax turns the row of scores into probabilities that sum to 1. Each token now has a weight distribution over itself + earlier tokens.
5. **`att @ v`** — weighted sum of value vectors using those weights. This is the actual information transfer.

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)   # concat the 12 heads back to 768
y = self.resid_dropout(self.c_proj(y))             # final linear projection
```
Glue the 12 heads' outputs back into a single 768-vector per token, then pass through one more linear layer (`c_proj`) to mix the heads' information together. Dropout (zeroing random activations during training) is regularization; with the default `dropout=0.0` it's a no-op.

**The causal mask is also why training is efficient:** because of masking, position 5's output never depends on position 6+, so the model can compute predictions for *all* positions in one forward pass and learn from every position simultaneously, while still respecting "no peeking ahead."

---

## 3. MLP (lines 78–92) — per-token thinking

```python
self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, ...)  # 768 -> 3072
self.gelu    = nn.GELU()
self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, ...)  # 3072 -> 768
```

After attention has gathered context, each token is processed **independently** by a small 2-layer neural net:
- Expand 768 → 3072 (4× wider),
- Apply **GELU** (a smooth nonlinearity, like a softer ReLU — without a nonlinearity here, stacking layers would collapse into a single linear function),
- Contract 3072 → 768.

**Intuition:** attention decides *what information to combine*; the MLP decides *what to do with it*. The wide middle layer gives the model capacity to compute richer per-token transformations (the bulk of the model's parameters actually live in these MLPs). A common mental model: this is where the network stores and applies "facts" and learned patterns.

---

## 4. Block (lines 94–106) — one transformer layer

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```

This combines the two pieces, and the structure here is subtle but important. Two things to notice:

**(a) The `x + ...` residual connections.** Instead of replacing `x`, each sub-layer computes an *update* that gets **added** to `x`. So information flows down the stack along a "residual highway," and each block only needs to learn a small adjustment. This is what makes 12+ layers trainable — gradients flow straight through the `+` without vanishing. Think of it as "x, but slightly improved by attention, then slightly improved by the MLP."

**(b) Pre-norm:** LayerNorm is applied to the *input* of each sub-layer (`self.attn(self.ln_1(x))`), not to the output. The residual path itself stays un-normalized. This (vs. GPT-1's post-norm) is what makes deep transformers train stably.

So one Block = "let tokens share context (attention), then let each token think (MLP), adding both as refinements." Stack 12 of these.

---

## 5. GPT (lines 118–193) — assembling the full model

### Construction (`__init__`)

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
    wpe = nn.Embedding(config.block_size, config.n_embd),   # position embeddings
    drop = nn.Dropout(config.dropout),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # the 12 blocks
    ln_f = LayerNorm(config.n_embd, bias=config.bias),      # final norm
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

- **`wte`** (token embedding): a lookup table mapping each of ~50k token IDs to a learned 768-vector. This is "what does this word mean."
- **`wpe`** (position embedding): a lookup table mapping each position 0–1023 to a learned 768-vector. Attention itself has no notion of order (it's a weighted sum — permuting inputs permutes outputs identically), so we **inject position information** by adding a position vector. This is "where in the sequence am I."
- **`h`**: the 12 Blocks.
- **`ln_f`**: a final LayerNorm before producing outputs.
- **`lm_head`**: maps each final 768-vector back to ~50k scores (logits) — one per vocabulary word — i.e. "how likely is each token to come next."

**Weight tying (line 138):**
```python
self.transformer.wte.weight = self.lm_head.weight
```
The input embedding table and the output projection **share the same weights**. Intuition: the matrix that maps "token → vector" is reused (transposed) for "vector → token scores." Saves ~38M parameters and tends to improve quality.

**Scaled init (lines 143–145):** the residual projection weights (`c_proj`) get initialized smaller, scaled by `1/sqrt(2·n_layer)`. Because residuals *add* across 12 layers, without this the activations would grow as they accumulate down the stack. A GPT-2 paper detail.

### Forward pass (lines 170–193)

```python
tok_emb = self.transformer.wte(idx)   # (b, t, 768)  look up each token
pos_emb = self.transformer.wpe(pos)   # (t, 768)     look up each position
x = self.transformer.drop(tok_emb + pos_emb)   # add them: "what" + "where"
for block in self.transformer.h:
    x = block(x)                       # 12 layers of attention+MLP
x = self.transformer.ln_f(x)
```

So the recipe is: **embed tokens, add positions, run through 12 blocks, final norm.** Then:

```python
if targets is not None:                      # TRAINING
    logits = self.lm_head(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
else:                                         # INFERENCE
    logits = self.lm_head(x[:, [-1], :])      # only the last position
    loss = None
```

- **Training:** compute logits at *every* position and measure cross-entropy loss against the true next tokens (`targets` is the input shifted by one). `ignore_index=-1` lets you mask out positions that shouldn't count. Because of causal masking, every position is a valid independent prediction, so one sequence gives you `T` training signals.
- **Inference:** you only care about predicting *after* the last token, so it skips computing logits for earlier positions — a small speedup.

---

## 6. The supporting methods (lower priority, but worth knowing)

These aren't part of the forward computation — skim them:

- **`generate` (305–330):** the autoregressive sampling loop. Forward the model → take last position's logits → divide by `temperature` (higher = more random) → optionally keep only `top_k` choices → softmax → sample one token → append → repeat. This is how you get text out.
- **`from_pretrained` (206–261):** loads OpenAI's GPT-2 weights from HuggingFace into this architecture. The fiddly part is that OpenAI used `Conv1D` layers whose weight matrices are transposed relative to `nn.Linear`, so certain weights get `.t()`'d during copy.
- **`configure_optimizers` (263–287):** sets up AdamW. The one idea worth absorbing: **2D+ tensors (matmul weights, embeddings) get weight decay; 1D tensors (biases, LayerNorm params) don't.** Decaying biases/norms hurts, so they're separated into groups.
- **`crop_block_size` (195–204):** surgically shrink the max sequence length (e.g. load GPT-2's 1024-context model but run it at 256) by trimming the position embeddings and mask.
- **`estimate_mfu` (289–303):** reports what fraction of an A100's theoretical FLOPS you're achieving — a hardware-efficiency diagnostic, irrelevant to model behavior.

---

## The one-paragraph summary

Tokens become 768-dim vectors (meaning + position). Twelve **Blocks** each refine those vectors in two steps: **attention** lets each token pull in information from earlier tokens (12 parallel heads, causally masked so no peeking ahead), and the **MLP** does per-token nonlinear processing. **Residual connections** (`x + ...`) and **LayerNorm** keep the deep stack trainable. A final linear layer turns each token's vector into next-token probabilities over the vocabulary. Training compares those predictions to the actual next tokens at every position at once; generation samples one token at a time and feeds it back.
