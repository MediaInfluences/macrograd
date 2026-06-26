# macrograd

A tiny **matrix-based autograd engine** — [micrograd](https://github.com/karpathy/micrograd), but operating on whole matrices instead of single scalars. Built from scratch to actually understand how backpropagation works under the hood.

> **Status:** Early WIP. The computational-graph skeleton and most forward ops are sketched in; the backward pass and a working training example are still to come. **Not yet runnable end-to-end.**

## Why "macro"?

Karpathy's micrograd builds an autograd engine over scalar `Value` objects. **macrograd scales that up to `Matrix` objects** — matmul, broadcasting, Hadamard (element-wise) ops, ReLU — which is much closer to how real frameworks like PyTorch actually compute. Same core idea (build a graph on the forward pass, walk it backward for gradients), bigger unit of work.

Learning project from [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).

## What's here

The `Matrix` class (`engine/matrix.py`) builds a computational graph as you compute. Each `Matrix` tracks its `elements`, a `grad`, its input nodes (`_inputs`), the op that produced it (`_op`), and a local `_backward` closure.

- **Forward ops sketched:** `transpose`, `hadamar_sum` (element-wise add), `hadamar_product` (element-wise mul), `__matmul__` (matrix multiply), `relu`, `max_margin_loss`
- **Broadcasting:** `_should_broadcast` (decides which side broadcasts) + `_broadcast`
- **`backwards()`:** topological sort of the graph + a reverse pass to propagate gradients

## What's not done yet

- Most per-op `_backward` gradient functions are still stubs (`pass`) — wiring up the backward pass is the next big push
- Broadcasting and a couple of forward ops (matmul indexing, the Hadamard helpers) have bugs to work through
- `backwards()` doesn't populate the topo order correctly yet
- No end-to-end training example (goal: fit something tiny) and no real test suite beyond a scratch `engine/test.py`

## Structure

```
macrograd/
├── engine/
│   ├── matrix.py     # the Matrix autograd engine (the heart of it)
│   └── test.py       # scratch test (transpose demo)
├── main.py           # entrypoint stub
├── pyproject.toml    # uv project, Python 3.13
└── README.md
```

## Running it

Uses [uv](https://docs.astral.sh/uv/) with Python 3.13.

```bash
uv run main.py
# poke at the engine directly:
uv run python -m engine.test
```

## Roadmap

- [ ] Finish `_backward` for every op (matmul, hadamard, relu, transpose, loss)
- [ ] Fix broadcasting + matmul indexing
- [ ] Make `backwards()` build the topological order correctly
- [ ] A minimal training loop — fit a tiny dataset end-to-end
- [ ] Gradient-correctness tests (compare against numerical gradients)
