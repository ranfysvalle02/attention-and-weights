# Demystifying the AI Black Box: Attention, Weights, Gradient Descent, and Softmax

> **Companion repo:** [`ranfysvalle02/attention-and-weights`](https://github.com/ranfysvalle02/attention-and-weights) — every claim in this post maps to runnable code:
> [`demo.py`](https://github.com/ranfysvalle02/attention-and-weights/blob/main/demo.py) (zero-dependency CLI walkthrough) ·
> [`app.py`](https://github.com/ranfysvalle02/attention-and-weights/blob/main/app.py) (FastAPI interactive playground) ·
> [`PIPELINE.md`](https://github.com/ranfysvalle02/attention-and-weights/blob/main/PIPELINE.md) (worked examples and diagrams).

If you have ever spent time interacting with a Large Language Model (LLM) and wondered, *"How does it actually know what to say?"* you are not alone. The inner workings of AI can feel like magic, but under the hood, it is a highly coordinated dance of mathematics.

When you dive into how LLMs work, you quickly run into a wall of intimidating jargon: **Attention, Weights, Gradient Descent, and Softmax**. It is incredibly common to get these concepts tangled up. Do they compete with each other? Are they doing the same thing?

Let's untangle the jargon and look at the exact mapping of how an AI learns, thinks, and answers.

---

### The Missing Link (Author's Note)

Here is the shortcut that finally clicked for me:

* **Gradient descent** is how the AI learns *over time*. **[TRAINING]**
* **Softmax** is how the AI chooses what to do *right now*. **[ATTENTION]**

For a long time I assumed **training did not use attention** — that attention was an inference-only trick layered on top of an already-trained model. That assumption is what kept the whole picture muddled.

The correction:

> At training time, **attention is what produces the context vector that the output weights learn to interpret.**

Attention is the **representation layer** that sits between raw tokens and the output decision. Training doesn't learn to read tokens directly — it learns to read attention's *output*. The **context vector** is the shared language between the transient (prompt-dependent) and permanent (weight-dependent) systems.

Once that landed, the rest of this article finally lined up. 

> **Latest demo (in this repo):** A pure-Python implementation of the attention mechanism, Q/K/V projections, and a gradient descent loop in [`demo.py`](https://github.com/ranfysvalle02/attention-and-weights/blob/main/demo.py) — using only the standard `math` library. It shows how changing **"fastest"** to **"safest"** alters the context vector and **flips the output (Redis vs. Postgres)** *without changing the frozen weights*. The single-word swap is enough to demonstrate how attention dynamically routes context by reshaping the question, not the knowledge. Prefer to *play* with it? Spin up the FastAPI companion in [`app.py`](https://github.com/ranfysvalle02/attention-and-weights/blob/main/app.py) and move the sliders yourself.

---

### The Brain's Memory: Model Weights

Before an AI can pay "attention" to anything, it needs a foundation of knowledge. This foundation lives in the model's **weights**.

You can think of weights as the model's crystallized memory of human language. They are simply massive grids of numbers (parameters) that represent the relationships between words and concepts. Everything the model "knows" from reading billions of pages of text is stored in these weights.

### The Dynamic Spotlight: The Attention Mechanism

A common misconception is that "attention" is just a static set of rules that filters your question. In reality, **attention is entirely driven by the model's weights.**

When you give an LLM a prompt, the attention mechanism acts as a dynamic spotlight. It uses specific sets of weights to evaluate your words through three different lenses:

* **Query:** What information am I looking for right now?
* **Key:** What information does this word contain?
* **Value:** If I decide this word is important, what is the actual content or semantic meaning that gets passed along?

By multiplying your words against these pre-existing weights, the model mathematically figures out the context. It realizes that in the sentence "The bark of the tree," the word "bark" should pay heavy attention to "tree" so it understands you mean wood, not a dog.

---

### The Teacher vs. The Balancer

So, how do those weights get there in the first place, and how does the model make its final choice on what word to output? This brings us to the two foundational mechanisms of machine learning: **Gradient Descent** and **Softmax**.

It is easy to view these as competing forces, but they actually do entirely different jobs at completely different times.

#### Gradient Descent: How the Model Learns

Gradient Descent is the "Teacher." It is the mathematical engine used exclusively during the **training phase**. It has absolutely nothing to do with generating an answer for you in real-time.

When a model is first built, its weights are random. During training, the model tries to predict the next word in a sentence, and it usually gets it wrong. The model uses a Loss Function to calculate exactly how wrong its guess was, and then Gradient Descent acts as the engine that works backward through the system to adjust those millions of weights based on that error.

Think of Gradient Descent as a hiker trying to walk to the bottom of a valley while blindfolded. It feels the slope of the hill and takes a step downward to reduce the model's error. Once the model is fully trained and the error is minimized, Gradient Descent's job is done.

> **The smallest possible training story.** Take a tiny JSON file containing one sentence — `"the fastest database is Redis"` — repeated twenty times. Each row produces the *same* context vector and the *same* error signal, and gradient descent nudges `W_out` in the *same* direction every time. After twenty repetitions, the model's probability for "Redis" goes from 50% to about 98%. The model didn't memorise a fact; it just let the same correction stack up. That's the entire mechanism, scaled. Real LLM "training on data" is the same loop with billions of varied examples and a much bigger weight space. See [`datasets/repeated_fastest.json`](https://github.com/ranfysvalle02/attention-and-weights/blob/main/datasets/repeated_fastest.json) and the **Training runs** picker in [`app.py`](https://github.com/ranfysvalle02/attention-and-weights/blob/main/app.py).

#### Softmax: How the Model Makes Decisions

If Gradient Descent is the teacher, Softmax is the "Balancer." Softmax is a mathematical function used constantly by the model, both during training and when answering your real-time prompts.

Neural networks produce raw, messy, unconstrained numbers — known in AI literature as **logits**. Softmax takes those messy logits and squashes them into a neat, clean set of percentages that always add up to 100%.

Softmax is used in two critical places:

* **Inside the Attention Spotlight:** It converts raw attention scores into percentages (e.g., deciding that "bark" should pay 85% attention to "tree" and 15% to "the").
* **At the Final Output:** When guessing the next word, it turns raw scores into probabilities (e.g., predicting an 80% chance the next word is "apple," and a 20% chance it is "orange").

---

### Tying It All Together: From Prompt to Answer

If we zoom out and look at the entire lifecycle of an LLM, the map from training to your screen looks like this:

1. **The Learning Phase:** During training, **Gradient Descent** systematically adjusts the model's **Weights** so that it stops making mistakes.
2. **The Spotlight Phase:** When you ask a question, the **Attention Mechanism** uses those static, learned weights to figure out which parts of your prompt matter the most.
3. **The Decision Phase:** The model routes this information through its network and uses **Softmax** to convert its final calculations into a probability distribution, allowing it to confidently choose the best possible word to output.

And that is how the black box works. No magic, no facts stored in a database—just a brilliantly engineered pipeline of weights, attention, learning, and probabilities working together to generate an answer, one word at a time.

---

### Run it yourself

Every concept above is mirrored in code you can read in one sitting:

* [`demo.py`](https://github.com/ranfysvalle02/attention-and-weights/blob/main/demo.py) — pure-Python inference + training, no dependencies. `python3 demo.py`.
* [`app.py`](https://github.com/ranfysvalle02/attention-and-weights/blob/main/app.py) — FastAPI app that renders this post alongside live attention bars, a softmax temperature slider, and a training loop. `pip install -r requirements.txt && python3 app.py`.
* [`PIPELINE.md`](https://github.com/ranfysvalle02/attention-and-weights/blob/main/PIPELINE.md) — the architectural deep-dive: pseudocode, hand-traced numbers, and mermaid diagrams mapping each step back to `demo.py`.

Full repo: [github.com/ranfysvalle02/attention-and-weights](https://github.com/ranfysvalle02/attention-and-weights).

---

### Next Steps: From Toy to Real

This repo is intentionally a *single head, single layer, hand-picked weights* demo. That's the right level of zoom to internalize the mechanics — but it's not where you stop. When you're ready to scale the same intuition up to a real model, these are the on-ramps in order of difficulty:

* **[Karpathy — *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)** — a two-hour video that turns the math in this post into a working character-level transformer. The natural next click after `demo.py`.
* **[Karpathy — *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)** — the full course. Starts at backprop on scalars and builds up to GPT. Every assumption this post hand-waves gets earned from first principles.
* **[`karpathy/minGPT`](https://github.com/karpathy/minGPT)** — ~300 lines of PyTorch implementing a real, trainable GPT. Same Q/K/V shapes you saw here, just stacked, multi-headed, and learned end-to-end.
* **[`karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT)** — the same idea, tuned for actually training on a GPU. The training script you'd hand a junior engineer.
* **[Jay Alammar — *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)** — the visual companion. If the math here clicked but the picture didn't, read this.
* **[Vaswani et al. — *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762)** — the original paper. Surprisingly readable once you've seen Q/K/V move with your own hands.

The jump from this repo to nanoGPT is: stack the attention block 6–12 times, use 64+ dimensions instead of 4, learn `W_Q`, `W_K`, `W_V` along with `W_OUTPUT` (not just the output projection), feed it a real corpus, and let gradient descent run for hours instead of 20 epochs. The mechanism doesn't change. Only the scale does.

----

### Appendix: Expanding on the AI Mechanics

**1. From Text to Math: Embeddings**
Before a model can apply its weights or pay attention to anything, it must first translate human language into a language it understands: math. When you type a prompt, the model converts each word (or token) into a dense array of numbers known as an **embedding**. This numerical format captures the semantic meaning of the word, acting as the crucial bridge between your text and the model's complex mathematical calculations.

**2. Controlling the Output: Softmax and Temperature**
When the **Softmax** function turns the model's final raw scores into probabilities (e.g., an 80% chance for "apple" and a 20% chance for "orange"), users can often influence how the model acts on those percentages using a setting called **Temperature**. A low temperature forces the model to almost always pick the safest, highest-probability word (making it predictable and focused), while a higher temperature encourages the model to occasionally risk choosing lower-probability words (resulting in more creative, varied outputs).

**3. Distinct Filters: Generating Query, Key, and Value**
While the Attention mechanism uses the model's weights to evaluate your words, it does so by creating three distinct representations for every token. The model mathematically multiplies your input prompt by three *different* sets of learned weights to generate the **Query**, **Key**, and **Value**. You can think of these separate weight matrices as three unique mathematical filters that transform the exact same input word into three distinct tools for the attention spotlight to use.
