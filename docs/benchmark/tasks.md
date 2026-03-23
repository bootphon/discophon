## Mapping units to phonemes

Let $\bm{u} = (u_1, \dots, u_T)$ be the sequence of discrete units to evaluate,
corresponding to a full dataset split, and let $\bm{p} = (p_1, \dots, p_T)$
be the corresponding sequence of gold phones, where $T$ denotes the number of
time steps at the resolution of the evaluated system. We denote by $\mathcal{P}$ the predefined set of phonemes,
and by $\mathcal{U}$ the one of units.
The empirical joint distribution of phones and units is

$$
\gdef\prob{\mathbb{P}}
\prob(i, j) = \frac{1}{T} \sum_{t=1}^T \left(p_t = i \wedge u_t = j\right), i \in \mathcal{P}, j \in \mathcal{U}.
$$

The many-to-one assignment maps each unit to its most frequent phoneme $A: j \mapsto \arg \max_{i \in \mathcal{P}} \prob(i, j)$.
The assigned sequence $\bm{a} = (\bm{a}_1, \dots, \bm{a}_T)$ is obtained by applying this mapping at each time step:
$\bm{a}_t = A(\bm{u}_t)$. For the one-to-one assignment,
we impose each phoneme to be mapped to a single unit: $A$ has to be a bijection.
We derive it by solving the linear assignment problem that maximizes $\prob(i, j)$ with SciPy.

Setting the vocabulary size is crucial for fair comparison: with this setup,
an unconstrained many-to-one mapping can be improved by increasing $|\mathcal{U}|$.
In the extreme case where $|\mathcal{U}| = T$ and where each unit appears exactly once, the mapping would be perfect.
A fixed vocabulary size eliminates this confound.


## Evaluation metrics

### Units quality

The PNMI between $\bm{p}$ and $\bm{u}$ is:

$$
\text{PNMI}(\bm{p}, \bm{u}) = \frac{I(\bm{p};\bm{u})}{H(\bm{p})} =
\frac{\sum_{i,j}\prob(i, j)\log \frac{\prob(i, j)}{\prob_{\bm{p}}(i) \prob_{\bm{u}}(j)}}{\sum_i \prob_{\bm{p}}(i) \log \prob_{\bm{p}}(i)},
$$

where $\prob_{\bm{p}}(i) = \sum_{j \in \mathcal{U}} \prob(i, j)$ and $\prob_{\bm{u}}(j) = \sum_{i \in \mathcal{P}} \prob(i, j)$
are the  marginal distributions. It measures the fraction of phone entropy explained by the discrete units.

### Recognition

PNMI is sensitive both to units' quality and their alignment.
Since the mapping produces a phone sequence $\bm{a}$ from the units $\bm{u}$,
we compare it directly to the gold sequence $\bm{p}$ by computing the **PER** to abstract away from the alignment.

### Segmentation

Recognition evaluates predicted labels but not their temporal alignment.
Segmentation evaluation is complementary: it ignores labels and only compares boundary positions.

We report $F_1$ and $\bm R$**-value**, which penalizes over-segmentation more than $F_1$.
We allow a tolerance of $\pm$20 ms around each gold boundary and split overlapping windows at their midpoint.

### Discriminability

We provide utilities to optionally compute ABX discriminability on continuous representations
of triphones or discrete units, using the [fastabx library](https://github.com/bootphon/fastabx).
ABX measures whether two instances of the same triphone (e.g., /bag/) are closer to one another
in embedding space than to instances of a minimally constrasting triphone (e.g., /beg/).
On discrete units, ABX is related to PNMI but with a hard threshold for success:
two realizations of the same triphone must be encoded as the same sequence.
On continuous representations, it is a useful proxy during development to guide
pretraining and select intermediate layers with easily available phonetic information without requiring discretization.
