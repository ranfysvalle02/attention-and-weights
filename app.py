"""FastAPI front-end for the attention-and-weights demo.

Serves blog.md as an interactive page backed by the same math used in demo.py:
embeddings, Q/K/V projections, scaled dot-product attention, softmax, and a
gradient-descent training loop on the output weights.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
BLOG_PATH = BASE_DIR / "blog.md"
DATASETS_DIR = BASE_DIR / "datasets"


# ---------------------------------------------------------------------------
# Math primitives (mirrors demo.py so the web demo stays a faithful twin)
# ---------------------------------------------------------------------------

EMBED_DIM = 4
HEAD_DIM = 4

TOKEN_EMBEDDINGS: dict[str, list[float]] = {
    "the":      [ 0.10, -0.05,  0.02,  0.08],
    "fastest":  [ 0.90,  0.05, -0.10,  0.20],
    "safest":   [ 0.05,  0.90, -0.10,  0.20],
    "best":     [ 0.50,  0.50,  0.10,  0.30],
    "database": [ 0.10,  0.10,  0.85,  0.60],
    "is":       [ 0.02,  0.02,  0.05,  0.03],
}

W_Q: list[list[float]] = [
    [1.5, 1.5, 2.0, 2.0],
    [1.5, 1.5, 2.0, 2.0],
    [1.0, 1.0, 2.0, 2.0],
    [1.5, 1.5, 2.0, 2.0],
]
W_K: list[list[float]] = [
    [2.0, 0.0, 1.0, 1.0],
    [0.0, 2.0, 1.0, 1.0],
    [0.5, 0.5, 2.0, 1.5],
    [0.5, 0.5, 1.0, 1.5],
]
W_V: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
W_OUTPUT: dict[str, list[float]] = {
    "Redis":    [5.0, 0.5, 3.0, 1.0],
    "Postgres": [0.5, 5.0, 3.0, 1.0],
}


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mat_vec(matrix: list[list[float]], vec: list[float]) -> list[float]:
    return [dot(row, vec) for row in matrix]


def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def vec_scale(vec: list[float], s: float) -> list[float]:
    return [x * s for x in vec]


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    t = max(temperature, 1e-6)
    scaled = [l / t for l in logits]
    mx = max(scaled)
    exps = [math.exp(l - mx) for l in scaled]
    s = sum(exps)
    return [e / s for e in exps]


def round_vec(vec: list[float], ndigits: int = 4) -> list[float]:
    return [round(x, ndigits) for x in vec]


def round_mat(mat: list[list[float]], ndigits: int = 4) -> list[list[float]]:
    return [round_vec(row, ndigits) for row in mat]


# ---------------------------------------------------------------------------
# Inference + training (returning structured JSON, not printing)
# ---------------------------------------------------------------------------

def _validate_matrix(name: str, mat: list[list[float]], rows: int, cols: int) -> list[list[float]]:
    if len(mat) != rows or any(len(row) != cols for row in mat):
        raise ValueError(f"{name} must be {rows}x{cols}")
    return [[float(x) for x in row] for row in mat]


def _validate_vec(name: str, vec: list[float], dim: int) -> list[float]:
    if len(vec) != dim:
        raise ValueError(f"{name} must have length {dim}")
    return [float(x) for x in vec]


def _resolve_weights(overrides: dict[str, Any] | None) -> tuple[
    dict[str, list[float]], list[list[float]], list[list[float]], list[list[float]], dict[str, list[float]]
]:
    """Return (embeddings, W_Q, W_K, W_V, W_out) with optional overrides applied."""

    embeddings = {t: list(v) for t, v in TOKEN_EMBEDDINGS.items()}
    wq = [list(row) for row in W_Q]
    wk = [list(row) for row in W_K]
    wv = [list(row) for row in W_V]
    wout = {c: list(v) for c, v in W_OUTPUT.items()}

    if not overrides:
        return embeddings, wq, wk, wv, wout

    embed_over = overrides.get("embeddings")
    if isinstance(embed_over, dict):
        for token, vec in embed_over.items():
            if token not in embeddings:
                raise ValueError(f"Unknown token in embeddings override: {token}")
            embeddings[token] = _validate_vec(f"embeddings[{token}]", vec, EMBED_DIM)

    if "W_Q" in overrides and overrides["W_Q"] is not None:
        wq = _validate_matrix("W_Q", overrides["W_Q"], HEAD_DIM, EMBED_DIM)
    if "W_K" in overrides and overrides["W_K"] is not None:
        wk = _validate_matrix("W_K", overrides["W_K"], HEAD_DIM, EMBED_DIM)
    if "W_V" in overrides and overrides["W_V"] is not None:
        wv = _validate_matrix("W_V", overrides["W_V"], HEAD_DIM, EMBED_DIM)

    wout_over = overrides.get("W_OUTPUT")
    if isinstance(wout_over, dict):
        for candidate, vec in wout_over.items():
            if candidate not in wout:
                raise ValueError(f"Unknown candidate in W_OUTPUT override: {candidate}")
            wout[candidate] = _validate_vec(f"W_OUTPUT[{candidate}]", vec, HEAD_DIM)

    return embeddings, wq, wk, wv, wout


def run_inference(
    prompt_tokens: list[str],
    *,
    attention_temperature: float = 1.0,
    output_temperature: float = 1.0,
    weight_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full pipeline returning every intermediate quantity for the UI."""

    embeddings, wq, wk, wv, wout = _resolve_weights(weight_overrides)

    unknown = [t for t in prompt_tokens if t not in embeddings]
    if unknown:
        raise ValueError(f"Unknown tokens: {unknown}")

    token_vecs = [embeddings[t] for t in prompt_tokens]

    queries = [mat_vec(wq, t) for t in token_vecs]
    keys = [mat_vec(wk, t) for t in token_vecs]
    values = [mat_vec(wv, t) for t in token_vecs]

    last_q = queries[-1]
    scale = math.sqrt(HEAD_DIM)
    raw_scores = [dot(last_q, k) / scale for k in keys]
    attn_weights = softmax(raw_scores, temperature=attention_temperature)

    context = [0.0] * HEAD_DIM
    for w, v in zip(attn_weights, values):
        context = vec_add(context, vec_scale(v, w))

    candidates = list(wout.keys())
    logits = [dot(context, wout[c]) for c in candidates]
    probs = softmax(logits, temperature=output_temperature)
    predicted_idx = max(range(len(probs)), key=lambda i: probs[i])

    return {
        "prompt": " ".join(prompt_tokens),
        "tokens": prompt_tokens,
        "embeddings": [round_vec(v) for v in token_vecs],
        "queries": [round_vec(v) for v in queries],
        "keys": [round_vec(v) for v in keys],
        "values": [round_vec(v) for v in values],
        "raw_scores": round_vec(raw_scores),
        "attention_weights": round_vec(attn_weights),
        "context": round_vec(context),
        "candidates": candidates,
        "logits": round_vec(logits),
        "probabilities": round_vec(probs),
        "predicted_token": candidates[predicted_idx],
        "attention_temperature": attention_temperature,
        "output_temperature": output_temperature,
        "weights_overridden": bool(weight_overrides),
    }


def train_output_weights(
    prompt_tokens: list[str],
    target: str,
    *,
    epochs: int = 20,
    learning_rate: float = 5.0,
    weight_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gradient descent on W_OUTPUT only (attention weights frozen)."""

    embeddings, wq, wk, wv, wout = _resolve_weights(weight_overrides)

    if target not in wout:
        raise ValueError(f"Unknown target token: {target}")
    unknown = [t for t in prompt_tokens if t not in embeddings]
    if unknown:
        raise ValueError(f"Unknown tokens: {unknown}")

    token_vecs = [embeddings[t] for t in prompt_tokens]
    queries = [mat_vec(wq, t) for t in token_vecs]
    keys = [mat_vec(wk, t) for t in token_vecs]
    values = [mat_vec(wv, t) for t in token_vecs]
    last_q = queries[-1]
    scale = math.sqrt(HEAD_DIM)
    raw_scores = [dot(last_q, k) / scale for k in keys]
    attn_weights = softmax(raw_scores)
    context = [0.0] * HEAD_DIM
    for w, v in zip(attn_weights, values):
        context = vec_add(context, vec_scale(v, w))

    candidates = list(wout.keys())
    target_idx = candidates.index(target)
    n = len(candidates)
    W = [[0.0] * HEAD_DIM for _ in range(n)]

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        logits = [dot(context, W[i]) for i in range(n)]
        probs = softmax(logits)
        loss = -math.log(probs[target_idx] + 1e-12)

        for i in range(n):
            err = probs[i] - (1.0 if i == target_idx else 0.0)
            for j in range(HEAD_DIM):
                W[i][j] -= learning_rate * err * context[j]

        history.append({
            "epoch": epoch,
            "probabilities": round_vec(probs),
            "loss": round(loss, 6),
        })

    return {
        "prompt": " ".join(prompt_tokens),
        "target": target,
        "candidates": candidates,
        "context": round_vec(context),
        "attention_weights": round_vec(attn_weights),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "history": history,
        "learned_weights": {c: round_vec(W[i]) for i, c in enumerate(candidates)},
    }


# ---------------------------------------------------------------------------
# Dataset loading + repeated-example training
# ---------------------------------------------------------------------------

# The teaching point: the dataset is just a list of sentences. Each sentence is
# parsed as (prompt tokens..., target token). Repeating the same sentence many
# times is what we want — it shows that repeated examples create repeated error
# signals, and gradient descent turns those signals into W_OUTPUT rows that
# favor the repeated target.


def parse_example(text: str) -> tuple[list[str], str]:
    """Split a sentence into (prompt_tokens, target_token).

    The final whitespace-separated token is the label; everything before it is
    the prompt. Token strings are returned verbatim (case preserved) so the
    caller can validate them against the toy vocabulary / candidate set.
    """

    tokens = [tok for tok in text.strip().split() if tok]
    if len(tokens) < 2:
        raise ValueError(
            f"Example must have at least 2 tokens (prompt + target): {text!r}"
        )
    return tokens[:-1], tokens[-1]


def load_dataset(name: str) -> dict[str, Any]:
    """Load a JSON dataset from `datasets/<name>.json`.

    Returns the raw dict so callers can read `name`, `description`, and the
    `examples` list. Validation against the model vocabulary happens later, at
    training time, so the loader stays generic.
    """

    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", name):
        raise ValueError(f"Invalid dataset name: {name!r}")

    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "examples" not in data:
        raise ValueError(f"Dataset {name!r} must be an object with an 'examples' list")
    if not isinstance(data["examples"], list) or not data["examples"]:
        raise ValueError(f"Dataset {name!r} must contain a non-empty 'examples' list")
    return data


def list_datasets() -> list[dict[str, Any]]:
    """Return a small summary of every dataset on disk (used by the UI)."""

    out: list[dict[str, Any]] = []
    if not DATASETS_DIR.exists():
        return out
    for path in sorted(DATASETS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        examples = data.get("examples", []) if isinstance(data, dict) else []
        sample = examples[0]["text"] if examples and isinstance(examples[0], dict) else ""
        out.append({
            "name": path.stem,
            "description": data.get("description", "") if isinstance(data, dict) else "",
            "example_count": len(examples) if isinstance(examples, list) else 0,
            "sample_text": sample,
        })
    return out


def train_output_weights_from_examples(
    examples: list[dict[str, Any]],
    *,
    learning_rate: float = 5.0,
    weight_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train W_OUTPUT by stepping through a list of (prompt, target) examples.

    Each example performs exactly one gradient update against the cross-entropy
    loss for that example's target. Repeating the same example many times in
    the list is the whole point of this entry point: it shows that the same
    context vector + the same target produces the same error signal every
    time, and gradient descent accumulates those nudges into W_OUTPUT.

    Returns the same shape as `train_output_weights`, but with `history` indexed
    by *example number* instead of epoch, plus a few story-friendly fields.
    """

    if not examples:
        raise ValueError("examples must be a non-empty list")

    embeddings, wq, wk, wv, wout = _resolve_weights(weight_overrides)
    candidates = list(wout.keys())
    n_candidates = len(candidates)
    scale = math.sqrt(HEAD_DIM)

    # Parse + validate all examples up front so we fail loudly before training.
    parsed: list[tuple[list[str], str]] = []
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict) or "text" not in ex:
            raise ValueError(f"Example #{i} must be an object with a 'text' field")
        prompt_tokens, target = parse_example(ex["text"])
        unknown = [t for t in prompt_tokens if t not in embeddings]
        if unknown:
            raise ValueError(f"Example #{i}: unknown tokens {unknown}")
        if target not in wout:
            raise ValueError(f"Example #{i}: unknown target token {target!r}")
        parsed.append((prompt_tokens, target))

    # Pre-compute each unique context vector once. Attention is frozen during
    # this training loop, so a repeated example always produces the exact same
    # context vector — exactly the property that makes the repetition story
    # clean.
    context_cache: dict[tuple[str, ...], list[float]] = {}

    def context_for(prompt_tokens: list[str]) -> list[float]:
        key = tuple(prompt_tokens)
        if key in context_cache:
            return context_cache[key]
        token_vecs = [embeddings[t] for t in prompt_tokens]
        queries = [mat_vec(wq, t) for t in token_vecs]
        keys = [mat_vec(wk, t) for t in token_vecs]
        values = [mat_vec(wv, t) for t in token_vecs]
        last_q = queries[-1]
        raw_scores = [dot(last_q, k) / scale for k in keys]
        attn_weights = softmax(raw_scores)
        ctx = [0.0] * HEAD_DIM
        for w, v in zip(attn_weights, values):
            ctx = vec_add(ctx, vec_scale(v, w))
        context_cache[key] = ctx
        return ctx

    # W_OUTPUT starts at zero so the first example always sees a uniform
    # softmax — the cleanest possible "before learning" snapshot.
    W = [[0.0] * HEAD_DIM for _ in range(n_candidates)]

    history: list[dict[str, Any]] = []
    target_counts: dict[str, int] = {c: 0 for c in candidates}

    for step, (prompt_tokens, target) in enumerate(parsed, start=1):
        context = context_for(prompt_tokens)
        target_idx = candidates.index(target)
        target_counts[target] += 1

        # Record the state *before* this example's update so the UI can show
        # the model's prediction on the very same context it's about to learn
        # from.
        logits = [dot(context, W[i]) for i in range(n_candidates)]
        probs = softmax(logits)
        loss = -math.log(probs[target_idx] + 1e-12)

        history.append({
            "step": step,
            "prompt": " ".join(prompt_tokens),
            "target": target,
            "probabilities": round_vec(probs),
            "loss": round(loss, 6),
        })

        # Gradient step on this single example.
        for i in range(n_candidates):
            err = probs[i] - (1.0 if i == target_idx else 0.0)
            for j in range(HEAD_DIM):
                W[i][j] -= learning_rate * err * context[j]

    # Final probabilities (after the last gradient step) for the most-recent
    # example — the headline number for the "did repetition push the model
    # toward the target?" story.
    last_prompt, last_target = parsed[-1]
    last_context = context_for(last_prompt)
    final_logits = [dot(last_context, W[i]) for i in range(n_candidates)]
    final_probs = softmax(final_logits)

    return {
        "candidates": candidates,
        "example_count": len(parsed),
        "unique_prompts": sorted({" ".join(p) for p, _ in parsed}),
        "target_counts": target_counts,
        "learning_rate": learning_rate,
        "history": history,
        "learned_weights": {c: round_vec(W[i]) for i, c in enumerate(candidates)},
        "final": {
            "prompt": " ".join(last_prompt),
            "target": last_target,
            "probabilities": round_vec(final_probs),
            "loss": round(-math.log(final_probs[candidates.index(last_target)] + 1e-12), 6),
        },
    }


# ---------------------------------------------------------------------------
# Tiny markdown -> HTML renderer (avoids adding a dependency)
# ---------------------------------------------------------------------------

_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_AUTOLINK_PATTERN = re.compile(r"(?<!\]\()(?<!\"|>)\b(https?://[^\s<>\)]+)")

_INLINE_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"<em>\1</em>"),
    (re.compile(r"`([^`]+?)`"), r"<code>\1</code>"),
]


def _link_sub(match: re.Match[str]) -> str:
    text, href = match.group(1), match.group(2)
    rel = ' rel="noopener noreferrer"' if href.startswith(("http://", "https://")) else ""
    target = ' target="_blank"' if href.startswith(("http://", "https://")) else ""
    return f'<a href="{href}"{target}{rel}>{text}</a>'


def _render_inline(text: str) -> str:
    text = _LINK_PATTERN.sub(_link_sub, text)
    for pattern, replacement in _INLINE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def markdown_to_html(md: str) -> str:
    """Convert the subset of Markdown used by blog.md into HTML."""

    lines = md.splitlines()
    out: list[str] = []
    i = 0
    paragraph: list[str] = []
    list_items: list[str] = []
    quote_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            text = " ".join(paragraph).strip()
            if text:
                out.append(f"<p>{_render_inline(text)}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            out.append("<ul>")
            for item in list_items:
                out.append(f"  <li>{_render_inline(item)}</li>")
            out.append("</ul>")
            list_items.clear()

    def flush_quote() -> None:
        if quote_lines:
            text = " ".join(quote_lines).strip()
            if text:
                out.append(f"<blockquote><p>{_render_inline(text)}</p></blockquote>")
            quote_lines.clear()

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            flush_paragraph()
            flush_list()
            flush_quote()
            i += 1
            continue

        if re.match(r"^-{3,}$", line.strip()):
            flush_paragraph()
            flush_list()
            flush_quote()
            out.append("<hr />")
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            flush_quote()
            level = len(heading.group(1))
            text = _render_inline(heading.group(2).strip())
            slug = re.sub(r"[^a-z0-9]+", "-", heading.group(2).lower()).strip("-")
            out.append(f'<h{level} id="{slug}">{text}</h{level}>')
            i += 1
            continue

        quote_match = re.match(r"^>\s?(.*)$", line)
        if quote_match:
            flush_paragraph()
            flush_list()
            quote_lines.append(quote_match.group(1).strip())
            i += 1
            continue

        list_match = re.match(r"^\s*[\*\-]\s+(.*)$", line)
        if list_match:
            flush_paragraph()
            flush_quote()
            list_items.append(list_match.group(1).strip())
            i += 1
            continue

        flush_list()
        flush_quote()
        paragraph.append(line.strip())
        i += 1

    flush_paragraph()
    flush_list()
    flush_quote()
    return "\n".join(out)


def load_blog_html() -> str:
    if not BLOG_PATH.exists():
        return "<p><em>blog.md not found.</em></p>"
    return markdown_to_html(BLOG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Attention & Weights — Interactive Blog")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class WeightOverrides(BaseModel):
    embeddings: dict[str, list[float]] | None = None
    W_Q: list[list[float]] | None = None
    W_K: list[list[float]] | None = None
    W_V: list[list[float]] | None = None
    W_OUTPUT: dict[str, list[float]] | None = None


class InferRequest(BaseModel):
    tokens: list[str] = Field(..., min_length=1)
    attention_temperature: float = Field(1.0, gt=0.0, le=10.0)
    output_temperature: float = Field(1.0, gt=0.0, le=10.0)
    weight_overrides: WeightOverrides | None = None


class RuleDef(BaseModel):
    adjective: str
    target: str
    repetitions: int = Field(0, ge=0, le=100)


class TrainInlineRequest(BaseModel):
    """Beginner-friendly training endpoint.

    The UI sends the structured pieces of the sentences — adjective, target
    answer, and a repetition count for each rule — and the server assembles
    an interleaved dataset in memory. Interleaving prevents catastrophic
    forgetting that would happen if we trained 20x A then 20x B.
    """
    rules: list[RuleDef]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/article")
async def api_article() -> JSONResponse:
    return JSONResponse({"html": load_blog_html()})


@app.get("/api/config")
async def api_config() -> JSONResponse:
    return JSONResponse({
        "embed_dim": EMBED_DIM,
        "head_dim": HEAD_DIM,
        "vocab": sorted(TOKEN_EMBEDDINGS.keys()),
        "candidates": list(W_OUTPUT.keys()),
        "embeddings": {t: round_vec(v) for t, v in TOKEN_EMBEDDINGS.items()},
        "W_Q": round_mat(W_Q),
        "W_K": round_mat(W_K),
        "W_V": round_mat(W_V),
        "W_OUTPUT": {c: round_vec(v) for c, v in W_OUTPUT.items()},
        "prompts": [
            ["the", "fastest", "database", "is"],
            ["the", "safest", "database", "is"],
            ["the", "best", "database", "is"],
        ],
    })


@app.post("/api/infer")
async def api_infer(req: InferRequest) -> JSONResponse:
    overrides = req.weight_overrides.model_dump(exclude_none=True) if req.weight_overrides else None
    try:
        result = run_inference(
            req.tokens,
            attention_temperature=req.attention_temperature,
            output_temperature=req.output_temperature,
            weight_overrides=overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(result)


@app.post("/api/train_inline")
async def api_train_inline(req: TrainInlineRequest) -> JSONResponse:
    """Train W_out from a set of repeated example rules assembled in memory.

    Interleaving the rules (rather than training all of rule A then all of
    rule B) prevents the model from simply overwriting earlier facts.
    """
    examples: list[dict[str, str]] = []
    max_reps = max((r.repetitions for r in req.rules), default=0)
    for i in range(max_reps):
        for r in req.rules:
            if i < r.repetitions:
                examples.append({"text": f"the {r.adjective} database is {r.target}"})

    if not examples:
        raise HTTPException(status_code=400, detail="Total repetitions must be > 0")

    try:
        result = train_output_weights_from_examples(examples)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["builder"] = {"rules": [r.model_dump() for r in req.rules]}
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
