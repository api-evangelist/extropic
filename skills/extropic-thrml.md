---
name: thrml
description: Write correct, idiomatic THRML code, centered on Ising machines and how they are trained. THRML is Extropic's JAX-based GPU simulator for the block Gibbs sampling programs that run natively on Extropic's probabilistic computing hardware. Use this skill whenever the user mentions THRML, Extropic, probabilistic computers, energy-based models (EBMs), Ising / Boltzmann / spin models, block Gibbs sampling, or asks to sample from or train a probabilistic graphical model in JAX — even if they do not say "THRML" by name. Especially use it for training a Boltzmann machine / Ising model via contrastive (positive/negative phase) KL-gradient estimation, graph-coloring a model for parallel updates, or dropping below the Ising layer to custom factors and samplers.
---

# THRML

THRML simulates the block Gibbs sampling programs that run natively on Extropic's
probabilistic computing chips. It is built on JAX and works with `jit`, `vmap`, and
`grad` — with one caveat that bites in training loops: an `IsingEBM` is a pytree
whose leaves include `SpinNode` objects, so it cannot be passed as a traced argument
into a `jit`ed function (see "Static structure vs. traced arrays"). Binary spin models map most naturally onto
transistor hardware, so the **Ising model (a.k.a. Boltzmann machine)** is the central
target and the focus of this skill — both sampling from one and training one.

When implementing something, write the code. No emojis. When you explain a
sampling concept, tie it to the concrete THRML object that realizes it.

## Import rule

Top-level imports only. Never import from a submodule (`thrml.block_sampling`,
`thrml.models.ising`, etc.).

```python
from thrml import Block, SpinNode, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, IsingTrainingSpec
from thrml.models import estimate_kl_grad, hinton_init
```

Rule: `from thrml import X` for core objects, `from thrml.models import X` for
model classes, nothing deeper.

`thrml` does not depend on `networkx` or `optax`, yet examples here use both.
`pip install thrml` will not pull either in: `pip install networkx optax`.

## How the API composes

THRML is layered. The Ising wrappers sit on top and fill in everything below them,
so for the central use case you only ever touch the top two rows.

| Layer | Object | What you supply |
|-------|--------|-----------------|
| Model | `IsingEBM(nodes, edges, biases, weights, beta)` | the parameters `b`, `J`, `β` |
| Partition | `Block`s from a graph coloring | which nodes update in parallel |
| Program | `IsingSamplingProgram(model, free_blocks, clamped_blocks)` | (auto-built from the two above) |
| Run | `sample_states` / `estimate_kl_grad` under a `SamplingSchedule` | the schedule and initial state |

What the wrappers fill in, so you do not have to:

- An `IsingEBM` holds `b`, `J`, `β` and, energy `E(s) = −β(Σ bᵢsᵢ + Σ Jᵢⱼsᵢsⱼ)`,
  automatically emits the two factors that encode it (a bias factor and a pairwise
  factor) via its `.factors` property.
- An `IsingSamplingProgram` automatically builds the `BlockGibbsSpec` from your
  free/clamped blocks, attaches the correct spin Gibbs sampler
  (`P(sᵢ=1 | nb) = σ(2γ)`, `γ` = local field) to every free block, and wires in
  the model's factors.

So you never construct factors, interaction groups, or samplers by hand for an
Ising model. You drop to those lower layers only for non-Ising energies (custom
factors) or non-spin variables (custom samplers) — see "Going lower-level" below.

## The one rule about blocks

A `Block` is a list of nodes that update **simultaneously**, so no two nodes in a
block may be neighbors. Make one block per color of the interaction graph:

```python
coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")  # general graphs
free_blocks = [Block([n for n, c in coloring.items() if c == k])
               for k in range(max(coloring.values()) + 1)]
```

Use the cheapest valid coloring for the graph at hand. Bipartite graphs (grids, RBMs,
complete bipartite) take an exact 2-coloring from `nx.bipartite.color(G)`. For a
complete bipartite graph / RBM the two node sets *are* the coloring — skip the
coloring call and wrap each set directly: `free_blocks = [Block(visible), Block(latent)]`.
A non-minimal coloring (e.g. DSATUR on a bipartite graph) is still correct; it just
produces more blocks than necessary, which slows sampling without changing results.

Every node carries a `SpinNode` (binary, `{-1,+1}`, stored as `bool_` with
`True` = +1). Attach them when you build the graph:
`nx.relabel_nodes(G, {coord: SpinNode() for coord in G.nodes}, copy=False)`.

## Carry blocks of nodes — let THRML do the indexing

This is the single most important idiom for writing clean THRML code. A node is an
identity handle, not a position: `SpinNode()` is a plain object, and THRML keys all
of its internal bookkeeping off object identity (a `BlockSpec` maps each node to its
slot in the packed global state for you). So structure every program around node
objects and `Block`s of them, and never maintain your own parallel arrays of integer
indices as the working representation. Integer indices are fine momentarily to *pick*
nodes (e.g. `np.random.choice(len(nodes), k)`), but convert to node objects
immediately (`chosen = [nodes[i] for i in idx]`) and work with objects and sets after
that.

Derive every sub-structure by set operations on nodes, not by slicing index arrays:

```python
clamp_set = set().union(*[set(b.nodes) for b in clamp_blocks])      # union of node sets
free_blocks = [Block([n for n in blk.nodes if n not in clamp_set])  # a color block minus clamped nodes
               for blk in coloring.values()]
node_to_color = {n: name for name, blk in coloring.items() for n in blk.nodes}  # dict keyed by node
```

This is exactly how the visible/latent split and the positive/negative colorings in
the training section are built — by removing node objects from blocks, never by
tracking which integer index is data. The `ThrmlBM` Boltzmann-machine wrapper below is
the reference for this style: it carries named blocks (`input_blocks`, `output_blocks`,
color blocks) and derives every clamped/free decomposition purely by set difference on
nodes, so it never reconciles an index array against the model state.

**How results come back — the only "slicing" you do.**
`sample_states(key, program, schedule, init_free, clamp_data, nodes_to_sample)`
returns a list **parallel to `nodes_to_sample`**: one array per requested block, in
the order you passed them. You choose what to read out by passing the right blocks,
and you recover each block's result positionally from the returned list — you do not
index into a flat global state vector. Each returned array has a leading `n_samples`
axis (from the schedule), with a chain axis prepended when you `vmap`:

```python
states = sample_states(key, program, schedule, init_free, clamp_data, [out_block, aux_block])
final_out = states[0][-1]   # block 0, last sample  -> shape (len(out_block),)
aux_trace = states[1]       # block 1, full (n_samples, len(aux_block)) readout trace
```

To map names back to outputs, zip your block dict against the returned list rather
than indexing a flat state:

```python
output = {name: arr[-1] for name, arr in zip(output_blocks.keys(), states[:len(output_blocks)])}
```

If you ever truly need the explicit node→state mapping (rare), `get_node_locations`,
`from_global_state`, and `block_state_to_global` expose it — but reaching for them
usually means you are fighting the abstraction instead of carrying blocks of nodes.

### A worked wrapper: `ThrmlBM`

Extropic's `ThrmlBM` is the reference for this idiom, and the shape to copy when you
build any THRML model wrapper. Every structure it holds is a named `Block` of nodes (or
a dict of them) and every decomposition is a set operation on node objects — there is
not one integer index array in the whole class. The skeleton:

```python
class ThrmlBM(eqx.Module):
    nodes: list[SpinNode]                              # the node pool — static structure
    edges: list[tuple[SpinNode, SpinNode]]
    input_blocks:  dict[str, Block[SpinNode]]          # named ports, addressed by name not index
    output_blocks: dict[str, Block[SpinNode]]
    block_collection: dict[str, Block[SpinNode]]       # all named groupings, for clamp/readout lookup by name
    non_input_coloring: dict[str, Block[SpinNode]]     # color blocks with inputs removed
    positive_coloring:  dict[str, Block[SpinNode]]     # ...and outputs removed too
    default_gibbs_iters: int                           # schedule default when none is supplied

    def __init__(self, nodes, edges, block_collection, input_names, output_names, node_coloring, ...):
        self.input_blocks  = {n: block_collection[n] for n in input_names}
        self.output_blocks = {n: block_collection[n] for n in output_names}
        input_nodes  = set().union(*(set(b.nodes) for b in self.input_blocks.values()))
        output_nodes = set().union(*(set(b.nodes) for b in self.output_blocks.values()))
        # derive each restricted coloring by set difference on node objects, dropping empties
        self.non_input_coloring = {
            name: Block([n for n in b if n not in input_nodes])
            for name, b in node_coloring.items() if any(n not in input_nodes for n in b)}
        self.positive_coloring = {
            name: Block([n for n in b if n not in input_nodes | output_nodes])
            for name, b in node_coloring.items() if any(n not in input_nodes | output_nodes for n in b)}

    def sample(self, key, inputs, params, ...):
        # trainable arrays live in a separate `params` pytree; the model is rebuilt from it each call
        ising = IsingEBM(self.nodes, self.edges, params.biases, params.weights, jnp.array(1.0))
        free_blocks = list(self.non_input_coloring.values())                  # blocks, not indices
        program = IsingSamplingProgram(ising, free_blocks, list(self.input_blocks.values()))
        states  = sample_states(key, program, schedule, init, clamp_data, list(self.output_blocks.values()))
        return {name: arr[-1] for name, arr in zip(self.output_blocks, states)}  # zip names to results
```

The moving parts to carry into your own wrappers: the trainable arrays sit in a separate
`params` pytree (`weights`, `biases`) while the module holds only static structure; the
`IsingEBM` is rebuilt from `params` on every call; sample-time clamps extend the
free/clamped split by the same set difference; and results come back by zipping block
*names* against the returned list, never by indexing a flat state. Full source (internal):
`github.com/extropic-ai/torx · thermalizers/boltzmann/thrml_bm.py`.

## Static structure vs. traced arrays

"Carry blocks of nodes" is one face of a deeper rule that governs every THRML
program: **structure is static, arrays are dynamic.** Nodes, `Block`s, the graph,
the `SamplingSchedule`, and the assembled `IsingEBM` / program / `IsingTrainingSpec`
are Python-level objects fixed at trace time; only the arrays — `biases`, `weights`,
`beta`, the sampling state, and keys — flow through `jit` / `vmap` / `grad`.
(`SamplingSchedule` is a hashable dataclass; the programs and EBMs are `equinox.Module`s.)

**Use the `eqx.filter_*` transforms and you never have to think about it.**
`eqx.filter_jit` / `eqx.filter_vmap` / `eqx.filter_grad` treat every non-array leaf —
the `SpinNode` objects, blocks, schedule — as static automatically, so you can hand
them a whole `IsingEBM`, program, or spec as an ordinary argument. THRML's own
`hinton_init` is written exactly this way: `@eqx.filter_jit` over a function whose
argument is a full `IsingEBM`. Reach for `filter_jit` first. Plain `jax.jit` traces
only array leaves, so under it you must instead close over the structure and pass the
arrays alone — which is what the training step below does, since it is threading
`biases`/`weights` through optax regardless.

**Keys are arrays — thread them.** Split before you branch (`jax.random.split(key, n)`),
never reuse a key, and construct keys with the typed `jax.random.key(seed)`. When you
`vmap` N parallel chains, split exactly N keys and map over them alongside the
per-chain init arrays.

**Do not differentiate through the sampler.** Block Gibbs sampling draws discrete
spins, so the chain is not differentiable and `jax.grad(sample_states)` is meaningless
for training. Train with the contrastive Monte-Carlo estimator instead: `estimate_kl_grad`
returns `∂KL/∂θ` as a difference of *sampled moments*, with no backprop through the chain
(see "Training"). Autodiff has exactly one role here — differentiating the closed-form
energy of a tiny model to check that estimator (see "Testing what you build").

## Sampling from an Ising model

The canonical path, end to end. `vmap` runs parallel chains; `jit` compiles.

```python
import jax, jax.numpy as jnp, networkx as nx
from thrml import Block, SpinNode, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

G = nx.grid_graph(dim=(16, 16), periodic=False)
nx.relabel_nodes(G, {c: SpinNode() for c in G.nodes}, copy=False)
nodes, edges = list(G.nodes), list(G.edges)

bicol = nx.bipartite.color(G)
free_blocks = [Block([n for n, c in bicol.items() if c == k]) for k in (0, 1)]

key = jax.random.key(0)
key, kb, kw = jax.random.split(key, 3)
model = IsingEBM(nodes, edges,
                 jax.random.normal(kb, (len(nodes),)),   # biases  b
                 jax.random.normal(kw, (len(edges),)),   # weights J
                 jnp.array(1.0))                          # beta

program = IsingSamplingProgram(model, free_blocks, [])    # no clamped blocks
schedule = SamplingSchedule(n_warmup=100, n_samples=50, steps_per_sample=5)

n_chains = 64
key, ki = jax.random.split(key)
init_free = hinton_init(ki, model, free_blocks, (n_chains,))  # list, one (n_chains, block_len) array per block

run = jax.jit(jax.vmap(lambda init, k: sample_states(k, program, schedule, init, [], [Block(nodes)])))
samples = run(init_free, jax.random.split(key, n_chains))
# samples[0] has shape (n_chains, n_samples, n_nodes); map True->+1, False->-1 to read spins.
```

`sample_states(key, program, schedule, init_state_free, state_clamp, nodes_to_sample)`
returns a list parallel to `nodes_to_sample`; each element has shape
`(n_samples, block_len)`, and the `n_samples` axis is always present even when
`n_samples == 1`. `vmap` prepends a chain axis, giving `(n_chains, n_samples, block_len)`,
so the final sample per chain is `samples[0][:, -1, :]`. `vmap` maps over the leading
axis of every array in `init_state_free` and over the keys, so keep init arrays
`(n_chains, block_len)` and split the same number of keys.

## Training an Ising model (the core workflow)

Training a Boltzmann machine minimizes `KL(data ‖ model)`. The gradient is a
**contrast of two expectations** — one with the data clamped, one with the model
running free:

```
ΔW = −β (⟨sᵢsⱼ⟩₊ − ⟨sᵢsⱼ⟩₋)        Δb = −β (⟨sᵢ⟩₊ − ⟨sᵢ⟩₋)
```

- **Positive phase (`+`)**: clamp the visible/data nodes to a data batch, Gibbs-sample
  the latent nodes, accumulate moments. This needs the data-dependent statistics.
- **Negative phase (`−`)**: Gibbs-sample *all* nodes from the model, accumulate moments.
  This is the model's own expectation.

THRML packages both phases in `IsingTrainingSpec` and runs them with
`estimate_kl_grad`, which returns parameter gradients you feed to any optimizer.

Two structural facts drive the whole setup:

1. **You must have latent nodes.** Split `nodes` into visible (data) and latent. If
   every node is data, the positive phase has nothing to sample and training is degenerate.
2. **The two phases color differently.** The negative phase colors the *full* graph.
   The positive phase reuses that coloring but with the data nodes removed (they are
   clamped, not sampled).

The setup builds both colorings by set operations on node objects (per the idiom
above):

```python
import jax, jax.numpy as jnp, numpy as np, networkx as nx, optax
from thrml import Block, SpinNode, SamplingSchedule
from thrml.models import IsingEBM, IsingTrainingSpec, estimate_kl_grad, hinton_init

# RBM: fully connected bipartite graph, visible ↔ latent.
# For complete bipartite the two node sets ARE the coloring — skip DSATUR.
n_vis, n_lat = 784, 256
vis_nodes = [SpinNode() for _ in range(n_vis)]
lat_nodes = [SpinNode() for _ in range(n_lat)]
all_nodes = vis_nodes + lat_nodes

G = nx.complete_bipartite_graph(n_vis, n_lat)
G = nx.relabel_nodes(G, {i: vis_nodes[i] for i in range(n_vis)} |
                        {n_vis + j: lat_nodes[j] for j in range(n_lat)})
edges = list(G.edges())

# Negative phase: sample both blocks. Positive phase: clamp visible, sample latent only.
free_blocks    = [Block(vis_nodes), Block(lat_nodes)]
clamped_blocks = [Block(lat_nodes)]

# For a general (non-bipartite) graph replace the three lines above with DSATUR:
#   data_set       = set(vis_nodes)
#   coloring       = nx.coloring.greedy_color(G, strategy="DSATUR")
#   free_blocks    = [Block([n for n,c in coloring.items() if c==k])
#                     for k in range(max(coloring.values())+1)]
#   clamped_blocks = [Block([n for n in b.nodes if n not in data_set]) for b in free_blocks]
#   clamped_blocks = [b for b in clamped_blocks if b.nodes]

schedule = SamplingSchedule(5, 100, 5)
beta     = jnp.array(1.0)
```

**Data is a bool array** of shape `(batch, n_visible)`, with `True` = spin +1 and
`False` = −1. Binarize before passing, e.g. `data = (mnist_batch > 0.5)` for a
`(batch, 784)` array. (Float `{0,1}` happens to work because clamping casts it, but
bool is the canonical input.)

The step itself. Two facts make it non-obvious, both baked into the pattern below:

- **Under plain `jax.jit`, `IsingEBM` and `IsingTrainingSpec` cannot be traced
  arguments** — their pytree leaves include `SpinNode` objects, which JAX cannot trace.
  (`eqx.filter_jit` would treat those leaves as static and let you pass the objects in;
  here we pass arrays anyway, because optax operates on `biases`/`weights` directly.) So
  pass only the array parameters; rebuild the model *and* the spec inside the function,
  closing over the static `nodes`/`edges`/blocks. Rebuilding inside `jit` is free:
  construction runs once at trace time and the traced arrays flow through correctly.
- **`IsingTrainingSpec` captures the model**, and `IsingEBM` is immutable (each update
  is a new object), so the spec is parameter-dependent and must be rebuilt every step.
  Inside `jit` that is automatic; in an interpreted loop you rebuild both explicitly.

```python
key = jax.random.key(0); key, kb, kw = jax.random.split(key, 3)
biases  = jax.random.normal(kb, (len(all_nodes),))
weights = jax.random.normal(kw, (len(edges),))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init((biases, weights))

def train_step(biases, weights, opt_state, data, key):
    model = IsingEBM(all_nodes, edges, biases, weights, beta)
    spec  = IsingTrainingSpec(model, [Block(vis_nodes)], [],
                              clamped_blocks, free_blocks, schedule, schedule)

    k_pos, k_neg, k_grad = jax.random.split(key, 3)
    batch = data.shape[0]
    init_pos = hinton_init(k_pos, model, clamped_blocks, (1, batch))    # (1, batch, block_len) per block
    init_neg = hinton_init(k_neg, model, free_blocks,    (batch,))      # (batch, block_len) per block

    # estimate_kl_grad(key, spec, bias_nodes, weight_edges, data, conditioning_values,
    #                  init_positive, init_negative) -> grad_w, grad_b, pos_moments, neg_moments
    grad_w, grad_b, _, _ = estimate_kl_grad(
        k_grad, spec, all_nodes, edges, [data], [], init_pos, init_neg) # init_pos MUST NOT be []

    # estimate_kl_grad returns ∂KL/∂θ. optax.update() negates internally;
    # apply_updates ADDS the result — equivalent to params - lr*grad. Subtracting would ascend KL.
    updates, new_opt_state = optimizer.update((grad_b, grad_w), opt_state, (biases, weights))
    biases, weights = optax.apply_updates((biases, weights), updates)
    return biases, weights, new_opt_state

train_step = jax.jit(train_step)
# loop: biases, weights, opt_state = train_step(biases, weights, opt_state, data_batch, subkey)
```

Sign convention, since it is easy to get backwards: `estimate_kl_grad` returns the
*positive* KL gradient `∂KL/∂θ`. With optax you **add** the optimizer's output
(`apply_updates` adds, and `update` already negated), so the net effect descends KL.
Plain SGD instead **subtracts** the raw gradient: `weights = weights - lr * grad_w`,
`biases = biases - lr * grad_b`. Subtracting the optax updates, or adding the raw
gradient, ascends KL.

The two init shapes differ for a reason. The negative phase runs `batch` independent
model chains, so `init_neg` per block is `(batch, block_len)`. The positive phase runs
`n_chains_pos` latent-sampling chains *per data point*, so `init_pos` per block carries
an extra leading axis `(n_chains_pos, batch, block_len)`. `n_chains_pos = 1` is the
standard choice (one chain per data point); raising it to e.g. 4 averages more latent
samples per datum before computing the positive-phase moment, reducing gradient
variance at the cost of proportionally more compute. Passing `[]` for the positive
init is the recurring training bug — the positive phase has latents to sample, so it
needs one initialized array per positive-sampling block. `estimate_kl_grad` returns
`(grad_w, grad_b, pos_moments, neg_moments)` in that order.

## Going lower-level (only when Ising does not fit)

Reach below the Ising wrappers only for energies or variables it cannot express.
Confirm exact signatures in `references/source.md` before writing these.

- **Non-quadratic / categorical energies** — build factors directly:
  `SpinEBMFactor(node_groups, weights)` (any order: one block = biases, two = pairwise,
  three = triplets) or `CategoricalEBMFactor(node_groups, weights)`, then assemble with
  `FactorSamplingProgram(spec, samplers, factors, [])` over a `BlockGibbsSpec(free, clamped)`.
- **Non-spin variables** (e.g. continuous) — define an `AbstractNode` subclass, give it a
  shape/dtype in `BlockGibbsSpec(..., node_shape_dtypes={MyNode: jax.ShapeDtypeStruct(...)})`,
  and implement an `AbstractConditionalSampler` whose `sample(key, interactions, active_flags,
  states, sampler_state, output_sd)` returns `(new_state, new_sampler_state)`.
- **Mixed node types in one color** — blocks are single-type, so make one block per type and
  group them as a SuperBlock (a tuple) passed to `BlockGibbsSpec`; they share input state but
  use their own samplers.

### Writing a correct custom sampler

A sampler's `sample` / `compute_parameters` runs *inside* the traced Gibbs loop, so the
JAX rules apply in full:

- **Branch on type, never on values.** `isinstance(interaction, MyInteraction)` is a
  compile-time choice over the (static) interaction types a block sees — fine, and exactly
  how `SpinGibbsConditional` dispatches. Never write a Python `if` on an array value; use
  `jnp.where` / `lax.select` / `lax.cond`. Padded interaction slots are handled by
  *multiplying* by `active_flags` and summing (`jnp.sum(weights * active * ..., axis=-1)`),
  never by dropping terms — the `active` mask has shape `[n_nodes, max_appearances]`.
- **Reach for stable primitives.** Build parameters with `jax.nn.log_sigmoid`,
  `jax.nn.softplus`, `jax.nn.log_softmax`, or `jax.scipy.special.logsumexp` rather than raw
  `exp`/`log`, and sample categoricals straight from logits with `jax.random.categorical`
  (the built-in `SoftmaxConditional` does — do not softmax then sample). Spins arrive as
  `bool`; convert with `2 * s.astype(jnp.int8) - 1` before any arithmetic, as THRML does
  internally.
- **Extend, don't reimplement.** To add a new interaction type to a built-in sampler,
  subclass it, handle your interaction in `compute_parameters`, and delegate the rest to the
  parent (the parametric `compute_parameters` / `sample_given_parameters` split makes this
  clean):

```python
class ExtendedSpinGibbs(SpinGibbsConditional):
    def compute_parameters(self, key, interactions, active_flags, states, sampler_state, output_sd):
        field = jnp.zeros(output_sd.shape, dtype=float)
        rest_i, rest_a, rest_s = [], [], []
        for interaction, active, state in zip(interactions, active_flags, states):
            if isinstance(interaction, MyInteraction):              # the term this subclass adds
                spins = 2 * jnp.stack(state, -1).astype(jnp.int8) - 1   # bool -> {-1,+1} before any arithmetic
                field += jnp.sum(interaction.weights * active * jnp.prod(spins, -1), axis=-1)
            else:                                                   # hand the rest to the base sampler
                rest_i.append(interaction); rest_a.append(active); rest_s.append(state)
        base, _ = super().compute_parameters(key, rest_i, rest_a, rest_s, sampler_state, output_sd)
        return field + base, sampler_state                          # γ contributions add; P(s=1)=σ(2γ)
```

The sign of your contribution follows the local field `∂(−E)/∂sᵢ`, matching how the base
`γ` is built — get it backwards and the chain samples the wrong distribution silently.

## Testing what you build

Sampling code fails quietly — a wrong sign or a miscolored block still produces
plausible-looking spins — so validate against ground truth rather than eyeballing samples:

- **Exact enumeration (small graphs).** For ≲ 20 nodes, enumerate all `2ⁿ` states, score each
  with `model.energy(state, blocks)`, normalize `p ∝ exp(−E)`, and compare to the empirical
  histogram from a long chain (THRML's tests hold the max relative error under ~0.02).
- **Moment matching.** Compare empirical `⟨sᵢ⟩` and `⟨sᵢsⱼ⟩` against
  `estimate_moments(key, nodes, edges, program, schedule, init, clamp)` — the per-phase
  estimator that `estimate_kl_grad` is built from. Raise warm-up and thinning
  (`SamplingSchedule(n_warmup≳1000, …, steps_per_sample≳5)`) until they agree.
- **Gradient check.** Validate training by differentiating the *exact* KL of a tiny model —
  its closed-form energy summed over all `2ⁿ` states — with `eqx.filter_value_and_grad`, then
  confirm the Monte-Carlo `estimate_kl_grad` matches (THRML tests require < ~0.01). This
  differentiates the analytic energy, **not** the sampler.
- **`vmap` vs. loop.** A `vmap`ped run and a Python-loop run over the same keys must produce
  identical samples; this catches padding / `active_flags` bugs on heterogeneous graphs.

Two dtype guardrails that otherwise surface as a `RuntimeError`: spin **state** arrays must be
`bool` and categorical state `uint8` (the *parameters* `biases`/`weights` stay float), and wrap
optimizer updates in `with jax.numpy_dtype_promotion("standard"):` so `optax.update` does not
trip JAX's strict dtype promotion.

## References

`references/source.md` is the annotated library API (tests removed): every module prefaced by
a note on how it composes, plus four worked examples up front. It is the ground truth for any
exact signature, default, or numerically subtle update rule.

`references/jax_equinox.md` is the JAX/Equinox playbook for THRML — the static-vs-dynamic,
`filter_*`, PRNG, no-grad-through-the-sampler, custom-sampler, and testing rules distilled into
a checklist with pointers into the source. Load it when writing or reviewing THRML code that
trains a model or drops below the Ising layer.