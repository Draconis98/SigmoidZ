# SigmoidZ Theory Notes

This note records the working derivation behind SigmoidZ. The goal is not to prove that the architecture is better than RMSNorm or DyT. The goal is to make the modeling assumption explicit, derive the corresponding mean-field update, and map that update to Transformer components.

## 1. Starting Point

Probabilistic Transformer (PT) models contextual word representation with a dependency variable $H_i$ and latent labels $Z_i$. Under mean-field variational inference, the updates can be written in a vectorized form that resembles self-attention plus a feed-forward representation update.

The modification here is:

$$
\text{Original PT: } Z_i \text{ is one label from a set of size } d
$$

$$
\text{SigmoidZ: } Z_i \text{ is a binary vector in } \{0, 1\}^d
$$

Each hidden coordinate is now a Bernoulli latent variable. Its mean parameter is:

$$
q_{i,a} = Q(Z_{i,a} = 1)
$$

This makes the hidden representation interpretable as a vector of Bernoulli probabilities, or as a centered version $2q_i - 1$.

## 2. Binary MFVI Update

Use a unary potential for each active coordinate:

$$
\phi_u(Z_{i,a} = 1) = \exp(S_{i,a})
$$

$$
\phi_u(Z_{i,a} = 0) = 1
$$

Use pairwise potentials that couple $Z_{i,a}$ and $Z_{j,b}$ when token $j$ is the dependency head of token $i$.

For the simplest active-active coupling:

$$
\phi_t(H_i = j, Z_{i,a} = 1, Z_{j,b} = 1) = \exp(T_{a,b})
$$

$$
\phi_t(\text{other cases}) = 1
$$

The mean-field update for coordinate $a$ is:

$$
Q(Z_{i,a} = 1) \propto \exp(S_{i,a} + G_{i,a})
$$

$$
Q(Z_{i,a} = 0) \propto \exp(0)
$$

where $G_{i,a}$ is the expected pairwise contribution under the current distributions of dependency heads and neighboring binary variables.

Normalizing the two states gives:

$$
q_{i,a} = \operatorname{sigmoid}(S_{i,a} + G_{i,a})
$$

This is the central SigmoidZ update.

## 3. Vectorized Transformer Mapping

The dependency distribution $Q(H_i = j)$ maps naturally to attention:

$$
A = \operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d_{\text{head}}}}\right)
$$

The expected neighboring state maps to an attention aggregation:

$$
M = AV
$$

The binary MFVI update can then be written as:

$$
Z = \operatorname{sigmoid}(U(X) + P(M))
$$

$$
Y = O(2Z - 1)
$$

$U$ is the unary term, $P$ is the pairwise interaction term, and $O$ maps the centered Bernoulli mean back to the model width.

This gives the research variant:

$$
A = \operatorname{causal\_attention}(X)
$$

$$
Z = \operatorname{sigmoid}(\operatorname{unary}(X) + \operatorname{pairwise}(A))
$$

$$
Y = \operatorname{out}(2Z - 1)
$$

## 4. Conservative Variant

The conservative version only replaces normalization layers. It keeps the standard LLaMA-style block:

$$
X = X + \operatorname{Attention}(\operatorname{Norm}(X))
$$

$$
X = X + \operatorname{MLP}(\operatorname{Norm}(X))
$$

but uses:

$$
\operatorname{SigmoidZNorm}(X) = \gamma \bigl(2 \operatorname{sigmoid}(2 \alpha X + \beta) - 1\bigr) + \delta
$$

Here $\alpha$ is a learned scalar, $\beta$ is a learned per-channel logit bias, and $\gamma, \delta$ are learned affine parameters.

## 5. Relation to DyT

DyT uses:

$$
\operatorname{DyT}(X) = \gamma \tanh(\alpha X) + \delta
$$

Sigmoid and tanh are exactly related:

$$
\tanh(x) = 2 \operatorname{sigmoid}(2x) - 1
$$

Therefore, when $\beta = 0$, SigmoidZNorm is functionally equivalent to DyT:

$$
\operatorname{SigmoidZNorm}(X) = \gamma \tanh(\alpha X) + \delta
$$

The difference is the interpretation. DyT is motivated as a direct replacement for normalization. SigmoidZNorm is motivated as a centered Bernoulli mean-field update.

The extra $\beta$ term lets each channel learn a prior log-odds shift for its binary latent coordinate.

## 6. Stability Notes

The squashing function bounds the normalized representation before the affine output:

$$
2 \operatorname{sigmoid}(x) - 1 \in (-1, 1)
$$

The derivative is also bounded:

$$
\frac{d}{dx}\bigl[2 \operatorname{sigmoid}(2 \alpha x) - 1\bigr] \le \alpha
$$

This does not replace all optimization benefits of RMSNorm. It only gives a bounded local transform with a controllable slope. The residual path still carries unbounded activations, and optimizer settings remain important.

For LLMs, DyT reports that smaller initial slopes improve stability at larger scales, and that attention blocks benefit from larger initial slopes than FFN and final layers. SigmoidZ follows that convention with `alpha_attn` and `alpha_other`. These config values are initial values; the corresponding `alpha` tensors are learned during training.

## 7. What This Proves and Does Not Prove

This derivation proves that the binary latent-variable modification leads to a sigmoid mean-field update.

It also proves that the conservative SigmoidZNorm layer is exactly DyT when the logit bias is zero.

It does not prove that SigmoidZ will outperform RMSNorm or DyT in pretraining. That requires empirical training runs at the target scale.
