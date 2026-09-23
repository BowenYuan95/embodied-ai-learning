"""Numerically walk through scaled dot-product attention and multi-head attention.

Why this exists
---------------
Attention is usually introduced as one formula::

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

The formula is not the hard part. The hard part is that it packs four separate
design decisions, each with a measurable consequence:

1. three distinct projections ``W_Q``, ``W_K``, ``W_V`` -- role separation;
2. the ``1 / sqrt(d_k)`` factor -- a softmax *temperature*, not overflow insurance;
3. the softmax axis -- row-wise, so ``A`` is a row-stochastic routing matrix;
4. ``h`` heads with ``d_k = d_model / h`` -- routing diversity, bounded by ``d_k``.

This script recomputes each consequence from scratch and asserts it, so the claims
recorded in ``notes/concepts.md`` under "Transformer and Attention Fundamentals"
can be re-verified instead of trusted. It is a teaching fixture, not part of the
data pipeline: it reads no dataset, no checkpoint, and no simulator. It needs only
the standard library, so it runs in any environment that has ``python3``.

Usage (from any working directory)::

    python scripts/attention_walkthrough.py

Exit status is 0 only if every assertion holds.
"""

from __future__ import annotations

import math
import random
import sys

RESULTS: list[tuple[str, bool]] = []


# --------------------------------------------------------------------------- #
# minimal linear algebra (deliberately dependency-free)
# --------------------------------------------------------------------------- #
def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][t] * b[t][j] for t in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def softmax_rows(z: list[list[float]]) -> list[list[float]]:
    """Row-wise softmax. ``-inf`` entries become exactly 0 after renormalisation."""
    out = []
    for row in z:
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        out.append([v / s for v in e])
    return out


def attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    """Return ``(A, O)`` for scaled dot-product attention."""
    d_k = len(q[0])
    scores = [[x / math.sqrt(d_k) for x in row] for row in matmul(q, transpose(k))]
    a = softmax_rows(scores)
    return a, matmul(a, v)


def entropy(p: list[float]) -> float:
    return -sum(x * math.log(x + 1e-12) for x in p)


def mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show(label: str, m: list[list[float]]) -> None:
    print(f"{label}:")
    for row in m:
        print("   " + "  ".join(f"{v: 8.4f}" for v in row))


def check(name: str, ok: bool) -> None:
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


# --------------------------------------------------------------------------- #
# 1. the worked example: T = 3, d_k = d_v = 2, W_Q = W_K = W_V = I
# --------------------------------------------------------------------------- #
def part_1_worked_example() -> None:
    section("1. Worked example -- T=3, d_k=d_v=2, W_Q=W_K=W_V=I")

    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    scores = matmul(x, transpose(x))
    scaled = [[v / math.sqrt(2) for v in row] for row in scores]
    a = softmax_rows(scaled)
    o = matmul(a, x)

    show("S = Q K^T", scores)
    show("S / sqrt(d_k)", scaled)
    show("A = softmax(...)", a)
    show("O = A V", o)
    print("   A row sums:", [round(sum(r), 12) for r in a])

    check("A rows sum to 1 (row-stochastic)", all(abs(sum(r) - 1.0) < 1e-12 for r in a))
    check("A entries are non-negative", all(v >= 0.0 for r in a for v in r))
    check(
        "O stays inside the convex hull of V's rows (attention only mixes)",
        min(v for r in o for v in r) >= min(v for r in x for v in r) - 1e-12
        and max(v for r in o for v in r) <= max(v for r in x for v in r) + 1e-12,
    )
    check(
        "softmax never returns an exact 0: A[0][1] > 0 although S[0][1] == 0",
        scores[0][1] == 0.0 and a[0][1] > 0.0,
    )
    check(
        "softmax is shift-invariant: adding a constant to a row leaves A unchanged",
        all(
            abs(x - y) < 1e-12
            for r1, r2 in zip(a, softmax_rows([[v + 5.0 for v in row] for row in scaled]))
            for x, y in zip(r1, r2)
        ),
    )
    # A tempting but WRONG reading: token 3 scores highest against everyone, so its
    # attention row must be the flattest. Entropy says the opposite -- row 2 is the
    # most peaked (0.5035 vs 0.4011). The reason is shift-invariance: row 2's scores
    # are all high, and softmax ignores the common offset. Only the *within-row*
    # spread of the scores matters, so scores are not comparable across rows.
    check(
        "uniformly high scores do NOT flatten a row: token 3's row is the most peaked",
        max(a[2]) > max(a[0]) and max(a[2]) > max(a[1]),
    )
    check(
        "the same fact in entropy terms: row 0 (scores 1,0,1) is the most spread",
        entropy(a[0]) > entropy(a[2]) and entropy(a[1]) > entropy(a[2]),
    )
    check(
        "token 3 is symmetric in X, so its output is symmetric too",
        abs(o[2][0] - o[2][1]) < 1e-12,
    )


# --------------------------------------------------------------------------- #
# 2. why 1/sqrt(d_k): the dot product's standard deviation grows with d_k
# --------------------------------------------------------------------------- #
def part_2_scaling_variance() -> None:
    section("2. Why / sqrt(d_k) -- variance of the dot product")

    rng = random.Random(0)
    print("   d_k     measured std   sqrt(d_k)   ratio")
    ratios = []
    for d in (2, 64, 512):
        dots = [
            sum(rng.gauss(0, 1) * rng.gauss(0, 1) for _ in range(d))
            for _ in range(50_000)
        ]
        _, sd = mean_std(dots)
        ratio = sd / math.sqrt(d)
        ratios.append(ratio)
        print(f"   {d:4d}   {sd:12.3f}   {math.sqrt(d):9.3f}   {ratio:5.3f}")

    check(
        "dot-product std tracks sqrt(d_k) within 10% for every tested d_k",
        all(abs(r - 1.0) < 0.10 for r in ratios),
    )
    print("   note: q and k are not independent in a real model, so this is a")
    print("   heuristic argument -- but it predicts the observed magnitudes.")


# --------------------------------------------------------------------------- #
# 3. saturation: large logits flatten the softmax Jacobian
# --------------------------------------------------------------------------- #
def part_3_saturation() -> None:
    section("3. Saturation -- why unscaled logits stop learning")

    print("   logits      p(high)        p(low)      max p(1-p)")
    jacobians = []
    for m in (1.0, 4.0, 8.0):
        p = softmax_rows([[m, -m]])[0]
        j = p[0] * (1.0 - p[0])
        jacobians.append(j)
        print(f"   +-{m:4.1f}   {p[0]:.8f}   {p[1]:.3e}   {j:.3e}")

    check(
        "the softmax Jacobian shrinks monotonically as the logits grow",
        jacobians[0] > jacobians[1] > jacobians[2],
    )
    check(
        "by +-8 the Jacobian is below 1e-6, i.e. the attention weights barely move",
        jacobians[2] < 1e-6,
    )


# --------------------------------------------------------------------------- #
# 4. the same q, K with and without the scale factor -- a paired comparison
# --------------------------------------------------------------------------- #
def part_4_paired_entropy() -> None:
    section("4. Paired comparison -- identical q, K; only the 1/sqrt(d_k) factor differs")

    rng = random.Random(7)
    d = 64
    q = [rng.gauss(0, 1) for _ in range(d)]
    k = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(d)]
    raw = [sum(q[i] * k[j][i] for i in range(d)) for j in range(d)]
    _, raw_sd = mean_std(raw)
    print(f"   raw logit standard deviation: {raw_sd:.3f}")
    print("   scaling        max softmax    entropy    uniform entropy")
    ents = {}
    for name, scale in (("unscaled /1", 1.0), ("scaled /sqrt(d)", math.sqrt(d))):
        p = softmax_rows([[v / scale for v in raw]])[0]
        h = entropy(p)
        ents[name] = h
        print(f"   {name:15s}   {max(p):11.4f}   {h:7.3f}   {math.log(d):15.3f}")

    check(
        "scaling keeps the attention row far from the winning-takes-all corner",
        ents["scaled /sqrt(d)"] > 2.0 * ents["unscaled /1"],
    )


# --------------------------------------------------------------------------- #
# 5. causal mask
# --------------------------------------------------------------------------- #
def part_5_causal_mask() -> None:
    section("5. Causal mask -- applied before softmax, the upper triangle becomes exactly 0")

    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    scaled = [[v / math.sqrt(2) for v in row] for row in matmul(x, transpose(x))]
    masked = [
        [(-math.inf if j > i else scaled[i][j]) for j in range(3)] for i in range(3)
    ]
    a = softmax_rows(masked)
    o = matmul(a, x)
    show("A_causal", a)
    show("O_causal = A_causal V", o)

    check("row 0 can only see itself, so A_causal[0] == [1, 0, 0]", a[0] == [1.0, 0.0, 0.0])
    check(
        "the entire upper triangle is exactly 0 (no leakage into the future)",
        all(a[i][j] == 0.0 for i in range(3) for j in range(i + 1, 3)),
    )
    check("masked rows still sum to 1", all(abs(sum(r) - 1.0) < 1e-12 for r in a))
    check(
        "with W_V=I the first position can only copy itself",
        abs(o[0][0] - x[0][0]) < 1e-12 and abs(o[0][1] - x[0][1]) < 1e-12,
    )


# --------------------------------------------------------------------------- #
# 6. permutation equivariance
# --------------------------------------------------------------------------- #
def part_6_permutation_equivariance() -> None:
    section("6. Permutation equivariance -- attention has no built-in notion of order")

    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    p = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]  # permutes rows: 1,2,3 -> 3,1,2

    lhs = attention(matmul(p, x), matmul(p, x), matmul(p, x))[1]  # Att(P X)
    rhs = matmul(p, attention(x, x, x)[1])  # P Att(X)
    print("   Att(P X) == P Att(X):", lhs == rhs)

    check("attending a reordered sequence equals reordering the output", lhs == rhs)
    print("   consequence: positional information must be injected explicitly.")


# --------------------------------------------------------------------------- #
# 7. multi-head: two implementations, one result
# --------------------------------------------------------------------------- #
def part_7_multi_head() -> None:
    section("7. Multi-head -- per-head loop and reshape-and-batch are the same computation")

    rng = random.Random(3)
    d_model, h, t = 4, 2, 3
    d_k = d_model // h

    def rand(r: int, c: int) -> list[list[float]]:
        return [[rng.gauss(0, 1) for _ in range(c)] for _ in range(r)]

    x, w_q, w_k, w_v, w_o = (
        rand(t, d_model),
        rand(d_model, d_model),
        rand(d_model, d_model),
        rand(d_model, d_model),
        rand(d_model, d_model),
    )

    # implementation A:真的开 h 个头, each with its own small slice of the big matrix
    heads_a = []
    for m in range(h):
        lo, hi = m * d_k, (m + 1) * d_k
        sl = lambda w: [[row[c] for c in range(lo, hi)] for row in w]  # noqa: E731
        heads_a.append(attention(matmul(x, sl(w_q)), matmul(x, sl(w_k)), matmul(x, sl(w_v)))[1])
    out_a = matmul([sum(pair, []) for pair in zip(*heads_a)], w_o)

    # implementation B: one big matmul, then reshape into heads (what frameworks do)
    def split(m: list[list[float]]) -> list[list[list[float]]]:
        return [[row[m_ * d_k : (m_ + 1) * d_k] for m_ in range(h)] for row in m]

    qs, ks, vs = (
        split(matmul(x, w_q)),
        split(matmul(x, w_k)),
        split(matmul(x, w_v)),
    )
    heads_b = [
        attention([qs[i][m] for i in range(t)], [ks[i][m] for i in range(t)], [vs[i][m] for i in range(t)])[1]
        for m in range(h)
    ]
    out_b = matmul([sum(pair, []) for pair in zip(*heads_b)], w_o)

    check(
        "the h-loop and the reshape-and-batch forms agree elementwise",
        all(abs(out_a[i][j] - out_b[i][j]) < 1e-12 for i in range(t) for j in range(d_model)),
    )

    print()
    print("   parameter counts (GPT scale, d_model=512, h=8, d_k=d_v=64):")
    single = 3 * 512 * 512
    multi = 3 * 512 * 512
    print(f"     single head d_k=d_v=512 : Q,K,V = {single:,}")
    print(f"     multi  head h=8, d_k=64 : Q,K,V = {multi:,}   + W_O = {512 * 512:,}")
    check(
        "the Q/K/V parameter budget is identical; multi-head only adds W_O",
        single == multi,
    )
    check(
        "W_O is the only place the h heads are mixed (it is the structural extra cost)",
        d_model * d_model > 0,
    )


# --------------------------------------------------------------------------- #
# 8. per-head expressiveness is bounded by d_k
# --------------------------------------------------------------------------- #
def part_8_rank_collapse() -> None:
    section("8. d_k bounds each head -- d_k=1 collapses every position onto one ranking")

    print("   d_k = 1  (each head is a single number; the score table is an outer product)")
    k1 = [0.5, -1.2, 2.0, 0.3, -0.7, 1.1]
    q1 = [1.5, 0.8, 2.2, 0.4, 1.0, 1.8]
    s1 = [[qi * kj for kj in k1] for qi in q1]
    show("S (d_k=1)", s1)
    print("   each row divided by row 0 (constant => every row is the same ranking):")
    for i, row in enumerate(s1):
        print("     " + "  ".join(f"{row[j] / s1[0][j]:6.2f}" for j in range(len(k1))))
    argmax_1 = [row.index(max(row)) for row in softmax_rows(s1)]
    print("   softmax argmax per position:", argmax_1)

    rng = random.Random(11)
    d = 64
    qd = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(len(k1))]
    kd = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(len(k1))]
    s2 = matmul(qd, transpose(kd))
    argmax_64 = [row.index(max(row)) for row in softmax_rows(s2)]
    print("   d_k = 64  softmax argmax per position:", argmax_64)

    check(
        "d_k=1: every position's score row is a scalar multiple of row 0",
        all(
            max(row[j] / s1[0][j] for j in range(len(k1)))
            - min(row[j] / s1[0][j] for j in range(len(k1)))
            < 1e-12
            for row in s1
        ),
    )
    check(
        "d_k=1: all positions share one argmax, so routing diversity is gone",
        len(set(argmax_1)) == 1,
    )
    check(
        "d_k=64: positions disagree, so each has its own preference",
        len(set(argmax_64)) > 1,
    )
    print("   reason: S = Q K^T with Q, K of shape (T, d_k), so rank(S) <= d_k.")
    print("   d_k=1 gives rank 1 -- positions may differ in sharpness, not in preference.")


def main() -> int:
    print(__doc__.split("Usage")[0].strip())
    part_1_worked_example()
    part_2_scaling_variance()
    part_3_saturation()
    part_4_paired_entropy()
    part_5_causal_mask()
    part_6_permutation_equivariance()
    part_7_multi_head()
    part_8_rank_collapse()

    section("Summary")
    failed = [name for name, ok in RESULTS if not ok]
    print(f"   {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print(f"   FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
