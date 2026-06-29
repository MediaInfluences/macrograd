# macrograd

A tiny **matrix-based autograd engine** — [micrograd](https://github.com/karpathy/micrograd), but operating on whole matrices instead of single scalars. Built from scratch to actually understand how backpropagation works under the hood.

> **Status:** Working autograd engine. Forward **and** backward passes run end-to-end, and the gradients are **verified against numerical (finite-difference) gradients to ~3e-10** on a 2-layer MLP. No training loop yet — that's next.

## Why "macro"?

Karpathy's micrograd builds an autograd engine over scalar `Value` objects. **macrograd scales that up to `Matrix` objects** — matmul, broadcasting, Hadamard (element-wise) ops, ReLU — which is much closer to how real frameworks like PyTorch actually compute. Same core idea (build a graph on the forward pass, walk it backward for gradients), bigger unit of work. No numpy, no torch — just nested lists.

Learning project from [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).

## What's here

The `Matrix` class (`engine/matrix.py`) builds a computational graph as you compute. Each `Matrix` tracks its `elements`, a `grad` (itself a `Matrix`), its input nodes (`_inputs`), the op that produced it (`_op`), and a local `_backward` closure.

- **Forward ops:** `__matmul__` (matrix multiply), `transpose`, `hadamar_sum` (element-wise add), `hadamar_product` (element-wise mul), `relu`, `max_margin_loss`
- **Operator overloads:** `+` (`__add__`/`__radd__`), `*` (`__mul__`/`__rmul__`), `@` for matmul — write `a + c`, `a * b`, `a @ b` directly
- **Broadcasting:** `_should_broadcast` decides which side broadcasts; `_broadcast` expands along any size-1 axis (`(1,1)`, `(1,m)`, `(n,1)`), and its backward **sum-reduces** the gradient back along exactly those axes
- **Backward pass:** every op has a real `_backward`; `backwards()` builds the topological order, seeds `dL/dL = 1`, and walks the graph in reverse to accumulate gradients
- **Verified:** analytic gradients match finite-difference numerical gradients to a max error of **3.3e-10** across all parameters of a 2-layer MLP

## What's not done yet

- **No training loop / optimizer** — gradients are computed but nothing updates the weights with them yet. Fitting a tiny dataset end-to-end is the next milestone
- Only `relu` and `max_margin_loss` so far — more activations / losses would broaden it

## Structure

```
macrograd/
├── engine/
│   ├── matrix.py          # the Matrix autograd engine (the heart of it)
│   ├── mat_ops_test.py    # forward-op coverage: transpose, hadamard +/*, matmul, all broadcast shapes
│   └── backprop_test.py   # 2-layer MLP forward + backward
├── main.py                # entrypoint stub
├── pyproject.toml         # uv project, Python 3.13
└── README.md
```

## Running it

Uses [uv](https://docs.astral.sh/uv/) with Python 3.13.

```bash
# exercise the forward ops across broadcast shapes:
uv run python engine/mat_ops_test.py

# run a 2-layer MLP forward + backward and print gradients:
uv run python engine/backprop_test.py
```

## Roadmap

- [x] Fix matmul
- [x] Rework broadcasting to handle every shape (incl. scalar) + reduce on the backward
- [x] Finish `_backward` for every op (matmul, hadamard, relu, transpose, loss)
- [x] Get `backwards()` running end-to-end
- [x] Gradient-correctness check against numerical gradients
- [ ] A minimal training loop — fit a tiny dataset end-to-end
- [ ] More ops / activations
