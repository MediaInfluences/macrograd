# macrograd

A tiny **matrix-based autograd engine** (and a small neural-net library on top of it) — [micrograd](https://github.com/karpathy/micrograd), but operating on whole matrices instead of single scalars. Built from scratch to actually understand how backpropagation works under the hood.

> **Status:** Working end-to-end. The autograd engine's gradients are **verified against numerical (finite-difference) gradients to ~3e-10**, and a multi-layer perceptron built on top **trains to zero loss** via forward → backprop → SGD. No numpy, no torch — just nested lists.

## Why "macro"?

Karpathy's micrograd builds an autograd engine over scalar `Value` objects. **macrograd scales that up to `Matrix` objects** — matmul, broadcasting, Hadamard (element-wise) ops, ReLU — which is much closer to how real frameworks like PyTorch actually compute. Same core idea (build a graph on the forward pass, walk it backward for gradients), bigger unit of work.

Learning project from [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).

## The engine

The `Matrix` class (`engine/matrix.py`) builds a computational graph as you compute. Each `Matrix` tracks its `elements`, a `grad` (itself a `Matrix`), its input nodes (`_inputs`), the op that produced it (`_op`), and a local `_backward` closure.

- **Forward ops:** `__matmul__` (matrix multiply), `transpose`, `hadamar_sum` (element-wise add), `hadamar_product` (element-wise mul), `relu`, `max_margin_loss`
- **Operator overloads:** `+` (`__add__`/`__radd__`), `*` (`__mul__`/`__rmul__`), `@` for matmul — write `a + c`, `a * b`, `a @ b` directly
- **Broadcasting:** `_should_broadcast` decides which side broadcasts; `_broadcast` expands along any size-1 axis (`(1,1)`, `(1,m)`, `(n,1)`), and its backward **sum-reduces** the gradient back along exactly those axes
- **Backward pass:** every op has a real `_backward`; `backwards()` builds the topological order, seeds `dL/dL = 1`, and walks the graph in reverse to accumulate gradients (skipping non-grad constants via `has_grad`)
- **Verified:** analytic gradients match finite-difference numerical gradients to a max error of **3.3e-10** across all parameters of a 2-layer MLP

## The neural net

A small layer library (`nn/layers.py`) built entirely on the engine, with each piece as its own module (like PyTorch's separate `nn.Linear` and `nn.ReLU`):

- **`DenseLayer`** — a pure affine map `x @ W + b` (no activation baked in)
- **`ActivationFunc`** — a standalone ReLU layer, stacked *between* dense layers. The output layer has none, so it stays **linear** — outputs can go negative and gradients keep flowing (no dying-ReLU freeze)
- **`MLP`** — stacks the layers, runs `forward()`, takes the SGD weight step in `backward()`, and `epoch()` runs the training loop. The weight step updates the weights directly, keeping it out of the autograd graph
- **`LossFunc`** — thin wrapper over `max_margin_loss`

A 4-layer MLP in `nn/mlp_test.py` (`Dense → ReLU → Dense → ReLU → Dense → ReLU → Dense → loss`) fits a target end-to-end: loss drops from ~3 to **0.0** within ~10 epochs, with the output satisfying every max-margin constraint (correct sign and magnitude, including the negative targets).

## Visualizing the graph

`Matrix.backwards(show_graph=True)` renders the computation graph with [Graphviz](https://graphviz.org/) after the backward pass — one node per `Matrix`, showing its op, shape, and **forward | grad values side by side**, with the operations drawn as their own nodes in between. From the MLP, flip it on with the `_show_graph` flag:

```python
nn = MLP(x1, lr, True)     # renders + opens the graph each backward pass
```

Graphviz is imported lazily, so the engine only needs it when you actually turn this on (`uv add graphviz` + `brew install graphviz` for the `dot` binary). Best on small expressions — the full MLP renders as a hairball.

## What's not done yet

- **Single-example only** — trains on one input→target pair; no dataset iteration / batching yet
- Only `relu` and `max_margin_loss` so far — more activations / losses would broaden it
- **In progress — softmax + NLL classification loss.** The groundwork just landed and is *untested*: an `exp()` op, a `softmax()` stub, and new arithmetic overloads (`-`, `/`, `**`, unary negate). These are checkpointed WIP with known bugs — don't rely on anything past the verified op list above until the gradient check covers them

## Structure

```
macrograd/
├── engine/
│   ├── matrix.py          # the Matrix autograd engine (the heart of it)
│   ├── mat_ops_test.py    # forward-op coverage: transpose, hadamard +/*, matmul, all broadcast shapes
│   └── backprop_test.py   # 2-layer MLP forward + backward
├── nn/
│   ├── layers.py          # DenseLayer, ActivationFunc, MLP, LossFunc — the neural-net library
│   └── mlp_test.py        # 4-layer MLP that trains to zero loss
├── main.py                # entrypoint stub
├── pyproject.toml         # uv project, Python 3.13
└── README.md
```

## Running it

Uses [uv](https://docs.astral.sh/uv/) with Python 3.13. Run the `nn` tests as modules from the project root:

```bash
# train the MLP and watch the loss drop to 0:
uv run python -m nn.mlp_test

# exercise the engine's forward ops across broadcast shapes:
uv run python engine/mat_ops_test.py

# 2-layer forward + backward, prints gradients:
uv run python engine/backprop_test.py
```

## Roadmap

- [x] Fix matmul
- [x] Rework broadcasting to handle every shape (incl. scalar) + reduce on the backward
- [x] Finish `_backward` for every op (matmul, hadamard, relu, transpose, loss)
- [x] Get `backwards()` running end-to-end
- [x] Gradient-correctness check against numerical gradients
- [x] A minimal training loop — fit a tiny example end-to-end (MLP → 0 loss)
- [x] Graph visualization — `backwards(show_graph=True)` renders the graph (forward + grad per node)
- [ ] Softmax + negative-log-likelihood loss (`exp` landed, WIP — backward + stub still to fix)
- [ ] Fix & gradient-check the new arithmetic dunders (`-`, `/`, `**`, unary negate)
- [ ] Train over a real dataset (iteration / batching)
- [ ] PyTorch-style indexing (`m[i, j]`)
- [ ] More ops / activations
