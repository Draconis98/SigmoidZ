# SigmoidZ Theory Notes

This note records the working derivation behind SigmoidZ. The goal is not to
prove that the architecture is better than RMSNorm or DyT. The goal is to make
the modeling assumption explicit, derive the corresponding mean-field update,
and map that update to Transformer components.

## 1. Starting Point

Probabilistic Transformer (PT) models contextual word representation with a
dependency variable $H_i$ and latent labels $Z_i$. In the original categorical
view, $Z_i$ is one label from a set of size $d$, so the MFVI update normalizes
with a softmax over labels.

SigmoidZ changes the latent state to a binary vector:

$$
Z_i = (Z_{i,1}, \ldots, Z_{i,d}), \qquad Z_{i,a} \in \{0, 1\}.
$$

Each hidden coordinate is now a Bernoulli latent variable. Under a factorized
mean-field approximation,

$$
Q(H, Z) = \prod_i Q(H_i) \prod_{i,a} Q(Z_{i,a}),
$$

write:

$$
r_{i,j} = Q(H_i = j), \qquad q_{i,a} = Q(Z_{i,a} = 1).
$$

The hidden representation can be read either as probabilities $q_i$ or as a
centered Bernoulli mean:

$$
m_i = 2q_i - 1 \in (-1, 1)^d.
$$

The important change is:

$$
\text{categorical latent label} \Rightarrow \text{softmax update},
$$

$$
\text{factorized Bernoulli latent coordinates} \Rightarrow \text{sigmoid log-odds update}.
$$

## 2. Potentials

Use a unary log-potential for token $i$ and coordinate $a$:

$$
\log \phi_u(Z_{i,a} = z) = u_{i,a}(z).
$$

The unary log-odds are:

$$
s_{i,a} = u_{i,a}(1) - u_{i,a}(0).
$$

Use a dependency-conditioned pairwise log-potential between dependent token $i$
and head token $j$:

$$
\log \phi_p(H_i = j, Z_{i,a} = z, Z_{j,b} = z') = t_{a,b}(z, z').
$$

For coordinate $Z_{i,a}$, the only part that matters in the Bernoulli update is
the log-potential difference between setting the coordinate to 1 versus 0:

$$
\Delta t_{a,b}(z') = t_{a,b}(1, z') - t_{a,b}(0, z').
$$

The most compact active-active special case is:

$$
t_{a,b}(1, 1) = T_{a,b},
$$

$$
t_{a,b}(1, 0) = t_{a,b}(0, 1) = t_{a,b}(0, 0) = 0,
$$

which gives:

$$
\Delta t_{a,b}(1) = T_{a,b}, \qquad \Delta t_{a,b}(0) = 0.
$$

This active-active case is the simplest ablation. The current
`SigmoidZAttentionUpdate` implements the more general two-branch form by
aggregating both $q_{j,b}$ and $1-q_{j,b}$ messages.

## 3. MFVI Coordinate Update

The coordinate-wise MFVI optimum is:

$$
\log Q^*(Z_{i,a} = z) =
\mathbb{E}_{Q_{-i,a}}[\log p(H, Z \mid X)] + \text{const}.
$$

Therefore the Bernoulli log-odds are:

$$
\eta_{i,a}
= \log \frac{Q^*(Z_{i,a}=1)}{Q^*(Z_{i,a}=0)}
= \mathbb{E}_{Q_{-i,a}}
\left[
\log p(Z_{i,a}=1, H, Z_{-i,a}\mid X)
- \log p(Z_{i,a}=0, H, Z_{-i,a}\mid X)
\right].
$$

Substituting the unary and pairwise potentials gives the general update:

$$
\eta_{i,a}
= s_{i,a}
+ \sum_j r_{i,j} \sum_b
\left[
q_{j,b} \Delta t_{a,b}(1)
+ (1 - q_{j,b}) \Delta t_{a,b}(0)
\right].
$$

The normalized Bernoulli mean is:

$$
q_{i,a} = \sigma(\eta_{i,a}).
$$

This is the central SigmoidZ result. The categorical PT update uses a softmax
because exactly one label is active. After replacing $Z_i$ by $\{0,1\}^d$, each
coordinate has its own two-state log-odds, so the update is a sigmoid.

For the active-active potential, the update reduces to:

$$
\eta_{i,a}
= s_{i,a} + \sum_j r_{i,j} \sum_b q_{j,b} T_{a,b},
$$

$$
q_{i,a} = \sigma
\left(
s_{i,a} + \sum_j r_{i,j} \sum_b q_{j,b} T_{a,b}
\right).
$$

If the graph includes both parent-to-child and child-to-parent factors, another
incoming-message term can be added. For a decoder-only causal Transformer, the
practical mapping keeps the causal parent message: token $i$ attends only to
allowed previous tokens $j \le i$.

## 4. Vectorized Form

Let the dependency distribution $r_{i,j}$ be represented by causal attention:

$$
A = \operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_{\text{head}}}} + \operatorname{causal\_mask}
\right).
$$

Let the expected neighboring Bernoulli active and inactive states be aggregated
by attention:

$$
M_1 = A Q_Z, \qquad M_0 = A(1 - Q_Z).
$$

The MFVI update can then be written:

$$
\eta = \beta + U(X) + P_1(M_1) + P_0(M_0),
$$

$$
Z = \sigma(\eta),
$$

$$
Y = O(2Z - 1).
$$

Here:

- $\beta$ is a learned prior log-odds bias.
- $U(X)$ is the unary evidence for each Bernoulli coordinate.
- $P_1(M_1)$ is the active-neighbor pairwise contribution.
- $P_0(M_0)$ is the inactive-neighbor pairwise contribution.
- $O$ maps the centered Bernoulli mean back to the model width.

This gives the one-step MFVI Transformer block:

$$
\tilde X = \operatorname{Norm}(X),
$$

$$
A = \operatorname{causal\_attention}(\tilde X),
$$

$$
Q_Z = \sigma(W_z\tilde X + b_z),
$$

$$
M_1 = A Q_Z, \qquad M_0 = A(1 - Q_Z),
$$

$$
\eta = \beta + W_u \tilde X + W_{p1}M_1 + W_{p0}M_0,
$$

$$
Y = W_o \left(2\sigma(\eta) - 1\right),
$$

$$
X' = X + Y.
$$

Multi-head attention corresponds to multiple dependency distributions
$r^{(h)}_{i,j}$. Concatenating head messages and applying $W_p$ is a low-rank,
head-factorized approximation of the full pairwise tensor $T_{a,b}$.

## 5. Transformer Structure Implications

The binary MFVI derivation suggests three concrete architectural changes.

### Bernoulli attention update

Replace the standard attention value update:

$$
Y = W_o(AV)
$$

with a log-odds update:

$$
Y = W_o
\left(
2\sigma(\beta + W_u X + W_{p1} A Q_Z + W_{p0} A(1 - Q_Z)) - 1
\right).
$$

This is the `research` variant in the codebase:

```text
neighbor_state = sigmoid(neighbor_logits(x) + neighbor_logit_bias)
active_message = causal_attention_message(neighbor_state)
inactive_message = causal_attention_message(1 - neighbor_state)
logits = unary(x) + pairwise_active(active_message) + pairwise_inactive(inactive_message) + logit_bias
z = 2 * sigmoid(logits) - 1
out = output_projection(z)
```

The update keeps standard attention for the dependency distribution $Q(H_i)$,
but changes the hidden-state update from an unconstrained linear value mixture
to a bounded Bernoulli mean-field step.

### Conservative SigmoidZ normalization

A lower-risk approximation is to keep the standard Transformer block and only
replace normalization-like transforms with a centered Bernoulli parameterization:

$$
\operatorname{SigmoidZNorm}(X)
= \gamma \left(2\sigma(2\alpha X + \beta) - 1\right) + \delta.
$$

The block remains:

$$
X' = X + \operatorname{Attention}(\operatorname{SigmoidZNorm}(X)),
$$

$$
X'' = X' + \operatorname{MLP}(\operatorname{Norm}(X')).
$$

This does not implement the full pairwise MFVI update, but it preserves the
main Bernoulli interpretation at normalization sites and is easier to stabilize
at LLM scale.

### Optional iterative MFVI block

The one-step block above is the direct Transformer analogue. A closer inference
procedure would run a small number of inner MFVI iterations:

$$
q^{(t+1)} =
\sigma\left(\beta + U(X) + P(A V(q^{(t)}))\right).
$$

In practice this is more expensive and can be harder to train. Stacking
Transformer layers already provides repeated refinement, so the codebase uses
one MFVI-style update per layer.

## 6. Relation to DyT

DyT uses:

$$
\operatorname{DyT}(X) = \gamma \tanh(\alpha X) + \delta.
$$

Sigmoid and tanh are exactly related:

$$
\tanh(x) = 2\sigma(2x) - 1.
$$

Therefore, when $\beta = 0$, SigmoidZNorm is functionally equivalent to DyT:

$$
\operatorname{SigmoidZNorm}(X)
= \gamma \tanh(\alpha X) + \delta.
$$

The difference is the interpretation. DyT is motivated as a direct replacement
for normalization. SigmoidZNorm is motivated as a centered Bernoulli mean-field
update. The extra $\beta$ term lets each channel learn a prior log-odds shift
for its binary latent coordinate.

## 7. Stability Notes

The squashing function bounds the normalized representation before the affine
output:

$$
2\sigma(x) - 1 \in (-1, 1).
$$

The derivative is also bounded:

$$
\frac{d}{dx}\left[2\sigma(2\alpha x) - 1\right] \le \alpha.
$$

This does not replace all optimization benefits of RMSNorm. It only gives a
bounded local transform with a controllable slope. The residual path still
carries unbounded activations, and optimizer settings remain important.

For LLMs, DyT reports that smaller initial slopes improve stability at larger
scales, and that attention blocks benefit from larger initial slopes than FFN
and final layers. SigmoidZ follows that convention with `alpha_attn` and
`alpha_other`. These config values are initial values; the corresponding
`alpha` tensors are learned during training.

## 8. What This Proves and Does Not Prove

This derivation proves that changing PT from categorical $Z_i$ to
$Z_i \in \{0,1\}^d$ changes the coordinate update from softmax-normalized label
probabilities to sigmoid-normalized Bernoulli log-odds.

It also proves that the research attention update is a one-step MFVI analogue,
and that the conservative SigmoidZNorm layer is exactly DyT when the logit bias
is zero.

It does not prove that SigmoidZ will outperform RMSNorm or DyT in pretraining.
That requires empirical training runs at the target scale.
