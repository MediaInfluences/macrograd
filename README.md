# macrograd

A tiny **matrix-based autograd engine** — [micrograd](https://github.com/karpathy/micrograd), but operating on whole matrices instead of single scalars. Built from scratch to actually understand how backpropagation works under the hood.

> **Status: mid-rebuild.** The engine's core ops are still verified against
> numerical (finite-difference) gradients to ~3e-10, but the `nn/` layer
> library is removed for now — the training loop is moving into the engine
> tests while I work out softmax/cross-entropy and a proper no-grad story
> for weight updates. Second-epoch training currently crashes (known, see
> below). No numpy, no torch — just nested lists.

## Why "macro"?

Karpathy's micrograd builds an autograd engine over scalar `Value` objects. **macrograd scales that up to `Matrix` objects** — matmul, broadcasting, Hadamard (element-wise) ops, ReLU — which is much closer to how real frameworks like PyTorch actually compute. Same core idea (build a graph on the forward pass, walk it backward for gradients), bigger unit of work.

Learning project from [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).

## The engine

The `Matrix` class (`engine/matrix.py`) builds a computational graph as you compute. Each `Matrix` tracks its `elements`, a `grad` (itself a `Matrix`), its input nodes (`_inputs`), the op that produced it (`_op`), and a local `_backward` closure.

- **Forward ops:** `__matmul__` (matrix multiply), `transpose`, `hadamar_sum` (element-wise add), `hadamar_product` (element-wise mul), `exp`, `log`, `relu`, `max_margin_loss`, `cross_entropy_loss` (fused softmax + NLL, WIP)
- **Operator overloads:** `+`, `-`, `*`, `/`, `**`, unary `-`, and `@` for matmul — operands can be raw Python scalars, they get wrapped automatically
- **Broadcasting:** `_should_broadcast` decides which side broadcasts; `_broadcast` expands along any size-1 axis (`(1,1)`, `(1,m)`, `(n,1)`), and its backward **sum-reduces** the gradient back along exactly those axes
- **Backward pass:** every op has a real `_backward`; `backwards()` builds the topological order, seeds `dL/dL = 1`, and walks the graph in reverse to accumulate gradients (skipping non-grad constants via `has_grad`)
- **Verified:** analytic gradients match finite-difference numerical gradients to a max error of **3.3e-10** across all parameters of a 2-layer MLP

## Visualizing the graph

`Matrix.backwards(show_graph=True)` renders the computation graph with [Graphviz](https://graphviz.org/) after the backward pass — one node per `Matrix`, showing its op, shape, and **forward | grad values side by side**, with the operations drawn as their own nodes in between:

```python
loss.backwards(True)     # renders + opens the graph after the backward pass
```

Graphviz is imported lazily, so the engine only needs it when you actually turn this on (`uv add graphviz` + `brew install graphviz` for the `dot` binary). Best on small expressions — a full training-loop graph renders as a hairball.

## What's not done yet

- **Single-example only** — trains on one input→target pair; no dataset iteration / batching yet
- **In progress — cross-entropy training loop.** `cross_entropy_loss`
  (fused softmax + NLL) lands with a hand-derived backward, and
  `backprop_nll_test.py` trains one epoch end-to-end. The second epoch
  crashes — and the reason is instructive: `w -= lr * w.grad` is itself
  made of engine ops, so the *weight update gets recorded into the
  computation graph*. The next backward pass walks into last epoch's
  gradient history and hits constants that have no grad. This is exactly
  why PyTorch makes you wrap updates in `torch.no_grad()` / use
  `.detach()` — macrograd needs the same concept, which is the next thing
  I'm building
- The `nn/` layer library (DenseLayer / MLP) is gone — **on purpose**. The
  old design was drifting Keras-wards: stack some layer objects, call
  `epoch()`, and the actual forward/backward/update mechanics vanish
  behind the abstraction. I'd rather follow the PyTorch mentality — *you*
  write the forward pass and the training loop, and the library just
  hands you autograd and building blocks. (Yes, PyTorch offers
  `nn.Sequential` too, but the flexibility of explicit code is the point —
  and in a project whose whole reason to exist is *seeing* the process,
  I don't want to abstract it away behind simple layer calls, however
  simple it is right now.) It'll come back as thinner, PyTorch-flavored
  modules once the loss + no-grad machinery settles

## Structure

```
macrograd/
├── engine/
│   ├── matrix.py               # the Matrix autograd engine (the heart of it)
│   ├── mat_ops_test.py         # forward-op coverage: transpose, hadamard +/*, matmul, all broadcast shapes
│   ├── backprop_mml_test.py    # 2-layer net, max-margin loss: forward + backward
│   ├── backprop_nll_test.py    # cross-entropy training loop (WIP — crashes on epoch 2, see above)
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

# the WIP cross-entropy training loop (epoch 1 works, epoch 2 crashes — documented above):
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
- [ ] Softmax + NLL: forward ✓, one epoch trains — fix graph-recorded weight updates (no_grad/detach) so epoch 2+ works
- [ ] Gradient-check the new ops (`-`, `/`, `**`, `exp`, `log`, unary negate, cross-entropy) against numerical gradients
- [ ] Train over a real dataset (iteration / batching)
- [ ] PyTorch-style indexing (`m[i, j]`)
- [ ] More ops / activations
