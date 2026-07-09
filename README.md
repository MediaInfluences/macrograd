# macrograd

A tiny **matrix-based autograd engine** — [micrograd](https://github.com/karpathy/micrograd), but operating on whole matrices instead of single scalars. Built from scratch to actually understand how backpropagation works under the hood.

> **Status: mid-rebuild.** The engine's core ops are still verified against
> numerical (finite-difference) gradients to ~3e-10, and the cross-entropy
> training loop now trains end-to-end (100 epochs) — the no-grad story for
> weight updates landed as a `Matrix.no_grad()` context manager. The `nn/`
> layer library is still removed for now while the new ops get
> gradient-checked. No numpy, no torch — just nested lists.

## Why "macro"?

Karpathy's micrograd builds an autograd engine over scalar `Value` objects. **macrograd scales that up to `Matrix` objects** — matmul, broadcasting, Hadamard (element-wise) ops, ReLU — which is much closer to how real frameworks like PyTorch actually compute. Same core idea (build a graph on the forward pass, walk it backward for gradients), bigger unit of work.

Learning project from [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).

## The engine

The `Matrix` class (`engine/matrix.py`) builds a computational graph as you compute. Each `Matrix` tracks its `elements`, a `grad` (itself a `Matrix`), its input nodes (`_inputs`), the op that produced it (`_op`), and a local `_backward` closure.

- **Forward ops:** `__matmul__` (matrix multiply), `transpose`, `hadamar_sum` (element-wise add), `hadamar_product` (element-wise mul), `exp`, `log`, `relu`, `max_margin_loss`, `cross_entropy_loss` (fused softmax + NLL, WIP)
- **Operator overloads:** `+`, `-`, `*`, `/`, `**`, unary `-`, and `@` for matmul — operands can be raw Python scalars, they get wrapped automatically
- **Broadcasting:** `_should_broadcast` decides which side broadcasts; `_broadcast` expands along any size-1 axis (`(1,1)`, `(1,m)`, `(n,1)`), and its backward **sum-reduces** the gradient back along exactly those axes
- **Backward pass:** every op has a real `_backward`; `backwards()` builds the topological order, seeds `dL/dL = 1`, and walks the graph in reverse to accumulate gradients (skipping non-grad constants via `has_grad`)
- **No-grad mode:** `with Matrix.no_grad():` disables graph recording, PyTorch-style — a class-level flag checked once in `__init__` means every `Matrix` built inside the block comes out as a leaf (no `_inputs`, no `_backward`). This exists because `w -= lr * w.grad` is itself made of engine ops: without it, every weight update got *recorded into the graph*, and the next epoch's backward walked into last epoch's history and crashed on grad-less constants — the same reason PyTorch makes you wrap updates in `torch.no_grad()`
- **Verified:** analytic gradients match finite-difference numerical gradients to a max error of **3.3e-10** across all parameters of a 2-layer MLP

## Visualizing the graph

`Matrix.backwards(show_graph=True)` renders the computation graph with [Graphviz](https://graphviz.org/) after the backward pass — one node per `Matrix`, showing its op, shape, and **forward | grad values side by side**, with the operations drawn as their own nodes in between:

```python
loss.backwards(True)     # renders + opens the graph after the backward pass
```

Graphviz is imported lazily, so the engine only needs it when you actually turn this on (`uv add graphviz` + `brew install graphviz` for the `dot` binary). Best on small expressions — a full training-loop graph renders as a hairball.

## What's not done yet

- **Single-example only** — trains on one input→target pair; no dataset iteration / batching yet
- **The new ops aren't gradient-checked yet** — `cross_entropy_loss` (fused
  softmax + NLL) has a hand-derived backward and the loss demonstrably
  falls over 100 epochs, but it (and `-`, `/`, `**`, `exp`, `log`) still
  need verification against finite-difference gradients like the core ops
- **No `detach()` yet** — `no_grad` covers the block-scoped case, but the
  per-tensor version (same values, cut from the graph) is still to come
- The `nn/` layer library (DenseLayer / MLP) is gone — **on purpose**. The
  old design was drifting Keras-wards: stack some layer objects, call
  `epoch()`, and the actual forward/backward/update mechanics vanish
  behind the abstraction. I'd rather follow the PyTorch mentality — *you*
  write the forward pass and the training loop, and the library just
  hands you autograd and building blocks. (Yes, PyTorch offers
  `nn.Sequential` too, but the flexibility of explicit code is the point —
  and in a project whose whole reason to exist is *seeing* the process,
  I don't want to abstract it away behind simple layer calls, however
  simple it is right now.) `no_grad` is what makes that trade viable:
  writing the update step yourself is only safe if you can step outside
  autograd to do it — a freedom an `epoch()` abstraction would never
  surface. It'll come back as thinner, PyTorch-flavored modules once
  the loss machinery is gradient-checked

## Structure

```
macrograd/
├── engine/
│   ├── matrix.py               # the Matrix autograd engine (the heart of it)
│   ├── mat_ops_test.py         # forward-op coverage: transpose, hadamard +/*, matmul, all broadcast shapes
│   ├── backprop_mml_test.py    # 2-layer net, max-margin loss: forward + backward
│   ├── backprop_nll_test.py    # cross-entropy training loop — 100 epochs, updates wrapped in no_grad
│   └── dunder_test.py          # exercises the arithmetic operator overloads
├── main.py                     # entrypoint stub
├── pyproject.toml              # uv project, Python 3.13
└── README.md
```

## Running it

Uses [uv](https://docs.astral.sh/uv/) with Python 3.13. Run the tests from `engine/`:

```bash
# exercise the engine's forward ops across broadcast shapes:
uv run python engine/mat_ops_test.py

# 2-layer forward + backward with max-margin loss, prints the loss:
cd engine && uv run python backprop_mml_test.py

# the cross-entropy training loop — 100 epochs, loss printed each step:
cd engine && uv run python backprop_nll_test.py
```

## Roadmap

- [x] Fix matmul
- [x] Rework broadcasting to handle every shape (incl. scalar) + reduce on the backward
- [x] Finish `_backward` for every op (matmul, hadamard, relu, transpose, loss)
- [x] Get `backwards()` running end-to-end
- [x] Gradient-correctness check against numerical gradients
- [x] A minimal training loop — fit a tiny example end-to-end (max-margin, single example)
- [x] Graph visualization — `backwards(show_graph=True)` renders the graph (forward + grad per node)
- [x] Dedup the topo sort so shared nodes don't double-count gradients
- [x] Softmax + NLL training loop — `Matrix.no_grad()` keeps weight updates out of the graph; 100 epochs, loss 3.51 → 0.66
- [ ] `detach()` — per-tensor graph cut-off to pair with the block-scoped `no_grad`
- [ ] Gradient-check the new ops (`-`, `/`, `**`, `exp`, `log`, unary negate, cross-entropy) against numerical gradients
- [ ] Train over a real dataset (iteration / batching)
- [ ] PyTorch-style indexing (`m[i, j]`)
- [ ] More ops / activations
