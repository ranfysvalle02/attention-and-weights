import math

# =====================================================================
# 1. MATHEMATICAL PRIMITIVES
# =====================================================================

def dot_product(vec1, vec2):
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

def softmax(logits, temperature=1.0):
    scaled = [l / max(temperature, 1e-6) for l in logits]
    mx = max(scaled)
    exps = [math.exp(l - mx) for l in scaled]
    s = sum(exps)
    return [e / s for e in exps]

def mat_vec(matrix, vec):
    return [dot_product(row, vec) for row in matrix]

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vec_scale(vec, s):
    return [x * s for x in vec]


# =====================================================================
# 2. TOKEN EMBEDDINGS (abstract 4-D — small enough to trace by hand)
# =====================================================================

EMBED_DIM = 4
HEAD_DIM = 4

TOKEN_EMBEDDINGS = {
    "the":      [ 0.10, -0.05,  0.02,  0.08],
    "fastest":  [ 0.90,  0.05, -0.10,  0.20],
    "safest":   [ 0.05,  0.90, -0.10,  0.20],
    "best":     [ 0.50,  0.50,  0.10,  0.30],
    "database": [ 0.10,  0.10,  0.85,  0.60],
    "is":       [ 0.02,  0.02,  0.05,  0.03],
}

# =====================================================================
# 3. LEARNED WEIGHT MATRICES (Q, K, V and output projection)
# =====================================================================

# These would be learned during training. We hand-set them so that:
#   - The prediction position ("is") produces a Query that aligns strongly
#     with Keys from content words (adjectives and nouns)
#   - V extracts a blend of adjective-meaning and noun-meaning
W_Q = [
    [ 1.5,  1.5,  2.0,  2.0],
    [ 1.5,  1.5,  2.0,  2.0],
    [ 1.0,  1.0,  2.0,  2.0],
    [ 1.5,  1.5,  2.0,  2.0],
]

W_K = [
    [ 2.0,  0.0,  1.0,  1.0],
    [ 0.0,  2.0,  1.0,  1.0],
    [ 0.5,  0.5,  2.0,  1.5],
    [ 0.5,  0.5,  1.0,  1.5],
]

W_V = [
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
]

# Output weights: one row per candidate answer token.
W_OUTPUT = {
    "Redis":    [ 5.0,  0.5,  3.0,  1.0],
    "Postgres": [ 0.5,  5.0,  3.0,  1.0],
}


# =====================================================================
# 4. ATTENTION MECHANISM
# =====================================================================

def attention(token_vecs, verbose=False):
    """Single-head self-attention over a sequence of token vectors.
    Returns the context vector at the LAST position (the prediction slot)."""

    n = len(token_vecs)

    # Project every token into Query, Key, Value space.
    queries = [mat_vec(W_Q, t) for t in token_vecs]
    keys    = [mat_vec(W_K, t) for t in token_vecs]
    values  = [mat_vec(W_V, t) for t in token_vecs]

    if verbose:
        print("\n  [Attention] Projections for each token:")
        for i, (q, k, v) in enumerate(zip(queries, keys, values)):
            print(f"    token {i}: Q={_fmt(q)}  K={_fmt(k)}  V={_fmt(v)}")

    # Compute attention from the LAST position to all positions.
    # This mirrors causal (autoregressive) prediction: the model attends
    # from the "is ___" slot backward over the full prompt.
    last_q = queries[-1]
    scale = math.sqrt(HEAD_DIM)
    scores = [dot_product(last_q, k) / scale for k in keys]
    weights = softmax(scores)

    if verbose:
        print(f"\n  [Attention] Scores from last position (scaled dot-product):")
        print(f"    raw:     {_fmt(scores)}")
        print(f"    softmax: {_fmt(weights)}")

    # Weighted sum of Value vectors = context vector.
    context = [0.0] * HEAD_DIM
    for w, v in zip(weights, values):
        context = vec_add(context, vec_scale(v, w))

    if verbose:
        print(f"    context: {_fmt(context)}")

    return context, weights


# =====================================================================
# 5. INFERENCE PIPELINE
# =====================================================================

def run_inference(prompt_tokens, verbose=True):
    """Full pipeline: embed -> attend -> score -> softmax -> prediction."""
    token_vecs = [TOKEN_EMBEDDINGS[t] for t in prompt_tokens]

    if verbose:
        print(f"\n  Prompt: \"{' '.join(prompt_tokens)}\"")
        print(f"  Token embeddings ({EMBED_DIM}-D):")
        for t, v in zip(prompt_tokens, token_vecs):
            print(f"    {t:<10} {_fmt(v)}")

    context, attn_weights = attention(token_vecs, verbose=verbose)

    # Score each candidate output token.
    options = list(W_OUTPUT.keys())
    logits = [dot_product(context, W_OUTPUT[opt]) for opt in options]

    if verbose:
        print(f"\n  [Output] Logits (dot of context with output weights):")
        for opt, score in zip(options, logits):
            print(f"    {opt:<10} {score:+.3f}")

    probs = softmax(logits)

    if verbose:
        print(f"\n  [Output] Probabilities (softmax):")
        for opt, p in zip(options, probs):
            print(f"    {opt:<10} {p:.2%}")

    return options, probs, attn_weights


# =====================================================================
# 6. TRAINING LOOP (gradient descent on output weights only)
# =====================================================================

def train_output_weights(prompt_tokens, target, epochs=20, lr=5.0):
    """Train W_OUTPUT from scratch using cross-entropy + gradient descent.
    Attention weights (W_Q, W_K, W_V) are held fixed for clarity."""

    token_vecs = [TOKEN_EMBEDDINGS[t] for t in prompt_tokens]
    context, _ = attention(token_vecs, verbose=False)

    options = list(W_OUTPUT.keys())
    target_idx = options.index(target)
    n_options = len(options)

    # Initialize output weights to zero.
    W = [[0.0] * HEAD_DIM for _ in range(n_options)]

    print(f"\n  Target: '{target}'")
    print(f"  Context vector: {_fmt(context)}")
    print(f"  {'Epoch':<6} | {' | '.join(f'P({o})' for o in options):<28} | Loss")
    print("  " + "-" * 60)

    for epoch in range(1, epochs + 1):
        logits = [dot_product(context, W[i]) for i in range(n_options)]
        probs = softmax(logits)
        loss = -math.log(probs[target_idx] + 1e-12)

        for i in range(n_options):
            err = probs[i] - (1.0 if i == target_idx else 0.0)
            for j in range(HEAD_DIM):
                W[i][j] -= lr * err * context[j]

        if epoch <= 5 or epoch % 5 == 0:
            prob_str = " | ".join(f"{p:.2%}" for p in probs)
            print(f"  #{epoch:<5} | {prob_str:<28} | {loss:.4f}")

    print("  " + "-" * 60)
    return {opt: W[i] for i, opt in enumerate(options)}


# =====================================================================
# 6b. REPEATED-EXAMPLE TRAINING (one gradient step per example)
# =====================================================================

def train_from_repeated_examples(examples, lr=5.0):
    """Train output weights by stepping once per example in a list.

    The pedagogical point: feeding the same sentence many times in a row is
    enough to push the output weights toward the repeated target. The same
    context vector produces the same error signal every time, and gradient
    descent stacks those nudges into a single direction.

    `examples` is a list of strings like "the fastest database is Redis".
    The final whitespace-separated token is treated as the label.
    """

    options = list(W_OUTPUT.keys())
    n_options = len(options)
    W = [[0.0] * HEAD_DIM for _ in range(n_options)]

    # Pre-compute every unique context vector once. Attention is frozen here,
    # so repeated prompts always produce the same context vector — which is
    # exactly the property that makes "repeat the same sentence" work as a
    # teaching tool.
    context_cache = {}

    def context_for(prompt_tokens):
        key = tuple(prompt_tokens)
        if key in context_cache:
            return context_cache[key]
        ctx, _ = attention([TOKEN_EMBEDDINGS[t] for t in prompt_tokens], verbose=False)
        context_cache[key] = ctx
        return ctx

    print(f"  {'Step':<6} | Example                          | "
          f"{' | '.join(f'P({o})' for o in options):<26} | Loss")
    print("  " + "-" * 90)

    parsed = []
    for raw in examples:
        toks = raw.strip().split()
        if len(toks) < 2:
            raise ValueError(f"Example too short: {raw!r}")
        parsed.append((toks[:-1], toks[-1]))

    snapshots = set()
    n = len(parsed)
    # Show a handful of evenly-spaced steps so the table fits on screen.
    show = {1, n}
    show.update({max(1, round(n * frac)) for frac in (0.1, 0.25, 0.5, 0.75)})

    for step, (prompt_tokens, target) in enumerate(parsed, start=1):
        context = context_for(prompt_tokens)
        target_idx = options.index(target)

        logits = [dot_product(context, W[i]) for i in range(n_options)]
        probs = softmax(logits)
        loss = -math.log(probs[target_idx] + 1e-12)

        if step in show and step not in snapshots:
            snapshots.add(step)
            prob_str = " | ".join(f"{p:.2%}" for p in probs)
            example_str = " ".join(prompt_tokens) + " -> " + target
            print(f"  #{step:<5} | {example_str:<32} | {prob_str:<26} | {loss:.4f}")

        for i in range(n_options):
            err = probs[i] - (1.0 if i == target_idx else 0.0)
            for j in range(HEAD_DIM):
                W[i][j] -= lr * err * context[j]

    print("  " + "-" * 90)
    return {opt: W[i] for i, opt in enumerate(options)}


def _load_dataset_examples(name):
    """Tiny JSON dataset loader, mirrored from app.py's loader.

    Demo.py stays dependency-free, so this uses only the standard library.
    """

    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "datasets", f"{name}.json")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex["text"] for ex in data["examples"]]


# =====================================================================
# 7. MAIN
# =====================================================================

def _fmt(vec):
    return "[" + ", ".join(f"{v:+.3f}" for v in vec) + "]"


if __name__ == "__main__":
    print("=" * 70)
    print("  SCENARIO A: \"the fastest database is ___\"")
    print("=" * 70)
    run_inference(["the", "fastest", "database", "is"])

    print("\n" + "=" * 70)
    print("  SCENARIO B: \"the safest database is ___\"")
    print("=" * 70)
    run_inference(["the", "safest", "database", "is"])

    print("\n" + "=" * 70)
    print("  SAME FROZEN WEIGHTS — DIFFERENT CONTEXT — DIFFERENT ANSWER")
    print("=" * 70)
    print("\n  The attention layer routed the prompt through different regions")
    print("  of the same frozen weight space. Nothing was updated.")

    # --- Training demo ---
    print("\n\n" + "=" * 70)
    print("  TRAINING: Teaching output weights from scratch")
    print("=" * 70)
    print("\n  Prompt: \"the best database is ___\"  Target: Postgres")
    W_learned = train_output_weights(["the", "best", "database", "is"], "Postgres")

    print("\n  Learned output weights:")
    for opt, w in W_learned.items():
        print(f"    W['{opt:<10}'] = {_fmt(w)}")

    # --- Repeated-example training demo ---
    print("\n\n" + "=" * 70)
    print("  TRAINING (from a JSON dataset): repeated examples push weights")
    print("=" * 70)
    print("\n  Dataset: datasets/repeated_fastest.json")
    print("  Each row is the same sentence: \"the fastest database is Redis\"")
    print("  One gradient step per row. No magic — just the same error signal,")
    print("  applied over and over.\n")
    examples = _load_dataset_examples("repeated_fastest")
    W_repeated = train_from_repeated_examples(examples)

    print("\n  Learned output weights after repetition:")
    for opt, w in W_repeated.items():
        print(f"    W['{opt:<10}'] = {_fmt(w)}")
"""
python3 demo.py 
======================================================================
  SCENARIO A: "the fastest database is ___"
======================================================================

  Prompt: "the fastest database is"
  Token embeddings (4-D):
    the        [+0.100, -0.050, +0.020, +0.080]
    fastest    [+0.900, +0.050, -0.100, +0.200]
    database   [+0.100, +0.100, +0.850, +0.600]
    is         [+0.020, +0.020, +0.050, +0.030]

  [Attention] Projections for each token:
    token 0: Q=[+0.275, +0.275, +0.250, +0.275]  K=[+0.300, +0.000, +0.185, +0.165]  V=[+0.100, -0.050, +0.020, +0.080]
    token 1: Q=[+1.625, +1.625, +1.150, +1.625]  K=[+1.900, +0.200, +0.575, +0.675]  V=[+0.900, +0.050, -0.100, +0.200]
    token 2: Q=[+3.200, +3.200, +3.100, +3.200]  K=[+1.650, +1.650, +2.700, +1.850]  V=[+0.100, +0.100, +0.850, +0.600]
    token 3: Q=[+0.220, +0.220, +0.200, +0.220]  K=[+0.120, +0.120, +0.165, +0.115]  V=[+0.020, +0.020, +0.050, +0.030]

  [Attention] Scores from last position (scaled dot-product):
    raw:     [+0.070, +0.363, +0.837, +0.056]
    softmax: [+0.182, +0.245, +0.393, +0.180]
    context: [+0.281, +0.046, +0.322, +0.305]

  [Output] Logits (dot of context with output weights):
    Redis      +2.701
    Postgres   +1.642

  [Output] Probabilities (softmax):
    Redis      74.25%
    Postgres   25.75%

======================================================================
  SCENARIO B: "the safest database is ___"
======================================================================

  Prompt: "the safest database is"
  Token embeddings (4-D):
    the        [+0.100, -0.050, +0.020, +0.080]
    safest     [+0.050, +0.900, -0.100, +0.200]
    database   [+0.100, +0.100, +0.850, +0.600]
    is         [+0.020, +0.020, +0.050, +0.030]

  [Attention] Projections for each token:
    token 0: Q=[+0.275, +0.275, +0.250, +0.275]  K=[+0.300, +0.000, +0.185, +0.165]  V=[+0.100, -0.050, +0.020, +0.080]
    token 1: Q=[+1.625, +1.625, +1.150, +1.625]  K=[+0.200, +1.900, +0.575, +0.675]  V=[+0.050, +0.900, -0.100, +0.200]
    token 2: Q=[+3.200, +3.200, +3.100, +3.200]  K=[+1.650, +1.650, +2.700, +1.850]  V=[+0.100, +0.100, +0.850, +0.600]
    token 3: Q=[+0.220, +0.220, +0.200, +0.220]  K=[+0.120, +0.120, +0.165, +0.115]  V=[+0.020, +0.020, +0.050, +0.030]

  [Attention] Scores from last position (scaled dot-product):
    raw:     [+0.070, +0.363, +0.837, +0.056]
    softmax: [+0.182, +0.245, +0.393, +0.180]
    context: [+0.073, +0.254, +0.322, +0.305]

  [Output] Logits (dot of context with output weights):
    Redis      +1.765
    Postgres   +2.578

  [Output] Probabilities (softmax):
    Redis      30.73%
    Postgres   69.27%

======================================================================
  SAME FROZEN WEIGHTS — DIFFERENT CONTEXT — DIFFERENT ANSWER
======================================================================

  The attention layer routed the prompt through different regions
  of the same frozen weight space. Nothing was updated.


======================================================================
  TRAINING: Teaching output weights from scratch
======================================================================

  Prompt: "the best database is ___"  Target: Postgres

  Target: 'Postgres'
  Context vector: [+0.198, +0.172, +0.359, +0.328]
  Epoch  | P(Redis) | P(Postgres)       | Loss
  ------------------------------------------------------------
  #1     | 50.00% | 50.00%              | 0.6931
  #2     | 17.89% | 82.11%              | 0.1971
  #3     | 11.21% | 88.79%              | 0.1189
  #4     | 8.23% | 91.77%               | 0.0859
  #5     | 6.53% | 93.47%               | 0.0675
  #10    | 3.23% | 96.77%               | 0.0328
  #15    | 2.15% | 97.85%               | 0.0218
  #20    | 1.62% | 98.38%               | 0.0163
  ------------------------------------------------------------

  Learned output weights:
    W['Redis     '] = [-1.348, -1.170, -2.448, -2.236]
    W['Postgres  '] = [+1.348, +1.170, +2.448, +2.236]
"""