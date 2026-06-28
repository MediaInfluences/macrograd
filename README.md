# macrograd

A tiny **matrix-based autograd engine** — [micrograd](https://github.com/karpathy/micrograd), but operating on whole matrices instead of single scalars. Built from scratch to actually understand how backpropagation works under the hood.

> **Status:** WIP. Forward pass is coming together — `__matmul__` and broadcasting now work and run cleanly in `engine/test.py`, and `Matrix` supports `+`/`*`/`@` operators. The **backward pass is still stubbed**, so it's not yet a working autograd engine end-to-end.

## Why "macro"?

Karpathy's micrograd builds an autograd engine over scalar `Value` objects. **macrograd scales that up to `Matrix` objects** — matmul, broadcasting, Hadamard (element-wise) ops, ReLU — which is much closer to how real frameworks like PyTorch actually compute. Same core idea (build a graph on the forward pass, walk it backward for gradients), bigger unit of work.

Learning project from [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).

## What's here

The `Matrix` class (`engine/matrix.py`) builds a computational graph as you compute. Each `Matrix` tracks its `elements`, a `grad`, its input nodes (`_inputs`), the op that produced it (`_op`), and a local `_backward` closure.

- **Forward ops working (tested):** `__matmul__` (matrix multiply) and broadcasting both run in `engine/test.py`
- **Operator overloads:** `+` (`__add__`/`__radd__`) and `*` (`__mul__`/`__rmul__`) wrap the Hadamard ops, plus `@` for matmul — so you can write `a + c`, `a @ b` directly
- **Forward ops present:** `transpose`, `hadamar_sum` (element-wise add), `hadamar_product` (element-wise mul), `relu`, `max_margin_loss`
- **Broadcasting:** `_should_broadcast` (decides which side broadcasts) + `_broadcast`, recently reworked after the first design was wrong
- **`backwards()`:** topological sort of the graph + a reverse pass to propagate gradients (not yet functional — see below)

## What's not done yet

- The backward pass isn't wired up yet — most per-op `_backward` functions are still stubs (`pass`), and `backwards()` itself needs finishing. This is the next big push
- Coverage is thin: the forward ops work in the demo, but the edge cases (odd shapes, non-square broadcasts) aren't exercised yet
- No end-to-end training example (goal: fit something tiny) and no real test suite beyond the scratch `engine/test.py`

## Structure

```
macrograd/
├── engine/
│   ├── matrix.py     # the Matrix autograd engine (the heart of it)
│   └── test.py       # scratch test (transpose, matmul, broadcast-add)
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
- [ ] Get `backwards()` running end-to-end
- [ ] Harden the forward ops against more shapes / edge cases
- [x] Fix matmul
- [x] Rework broadcasting after the first design was wrong
- [ ] A minimal training loop — fit a tiny dataset end-to-end
- [ ] Gradient-correctness tests (compare against numerical gradients)
