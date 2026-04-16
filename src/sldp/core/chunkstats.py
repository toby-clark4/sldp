import gc
from typing import cast

import numpy as np
import pandas as pd
import scipy.stats as st

from sldp.utils import fs


def _sum_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Return the elementwise sum of equally shaped arrays."""

    return np.sum(np.stack(arrays), axis=0)


# make idependent blocks
def collapse_to_chunks(
    ldblocks: pd.DataFrame,
    numerators: dict[int, np.ndarray],
    denominators: dict[int, np.ndarray],
    numblocks: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    """Collapse LD blocks into larger chunks for jackknife resampling."""

    # define endpoints of chunks
    ldblocks = ldblocks.copy()
    ldblocks["M_H"] = ldblocks.M_H.fillna(0)
    totalM = ldblocks.M_H.sum()
    chunksize = totalM / numblocks
    avgldblocksize = totalM / (ldblocks.M_H != 0).sum()
    chunkendpoints = [0]
    currldblock = 0
    currsize = 0
    while currldblock < len(ldblocks):
        while currsize <= max(1, chunksize - avgldblocksize / 2) and currldblock < len(ldblocks):
            currsize += ldblocks.iloc[currldblock].M_H
            currldblock += 1
        currsize = 0
        chunkendpoints += [currldblock]

    # store SNP indices of begin- and end-points of chunks
    chunkinfo = pd.DataFrame()

    # collapse data within chunks
    chunk_nums: list[np.ndarray] = []
    chunk_denoms: list[np.ndarray] = []
    chunkinfo_rows: list[dict[str, float | str]] = []
    for n, (i, j) in enumerate(zip(chunkendpoints[:-1], chunkendpoints[1:])):
        del n
        ldblock_ind = [ldblock for ldblock in ldblocks.iloc[i:j].index if ldblock in numerators.keys()]
        if len(ldblock_ind) > 0:
            chunk_nums.append(_sum_arrays([numerators[ldblock] for ldblock in ldblock_ind]))
            chunk_denoms.append(_sum_arrays([denominators[ldblock] for ldblock in ldblock_ind]))
            chunkinfo_rows.append(
                {
                    "ldblock_begin": min(ldblock_ind),
                    "ldblock_end": max(ldblock_ind) + 1,
                    "chr_begin": ldblocks.loc[min(ldblock_ind), "chr"],
                    "chr_end": ldblocks.loc[max(ldblock_ind), "chr"],
                    "bp_begin": ldblocks.loc[min(ldblock_ind), "start"],
                    "bp_end": ldblocks.loc[max(ldblock_ind), "end"],
                    "snpind_begin": ldblocks.loc[min(ldblock_ind), "snpind_begin"],
                    "snpind_end": ldblocks.loc[max(ldblock_ind), "snpind_end"],
                    "numsnps": sum(ldblocks.loc[ldblock_ind, "M_H"]),
                }
            )

    chunkinfo = pd.DataFrame(chunkinfo_rows)

    ## compute leave-one-out sums
    total_num = _sum_arrays(chunk_nums)
    total_denom = _sum_arrays(chunk_denoms)
    loonumerators = [total_num - chunk_num for chunk_num in chunk_nums]
    loodenominators = [total_denom - chunk_denom for chunk_denom in chunk_denoms]

    return chunk_nums, chunk_denoms, loonumerators, loodenominators, chunkinfo


# compute estimate of effect size
def get_est(num: np.ndarray, denom: np.ndarray, k: int, num_background: int) -> float:
    """Estimate the marginal annotation effect after background adjustment."""

    ind = list(range(num_background)) + [num_background + k]
    num = num[ind]
    denom = denom[ind][:, ind]
    try:
        return np.linalg.solve(denom, num)[-1]
    except np.linalg.LinAlgError:
        return np.nan


# compute standard error of estimate using jackknife
def jackknife_se(
    est: float,
    loonumerators: list[np.ndarray],
    loodenominators: list[np.ndarray],
    k: int,
    num_background: int,
) -> float:
    """Estimate the standard error of an effect size via jackknife."""

    m = np.ones(len(loonumerators))
    theta_notj = [get_est(nu, de, k, num_background) for nu, de in zip(loonumerators, loodenominators)]
    g = len(m)
    n = m.sum()
    h = n / m
    theta_J = g * est - ((n - m) / n * theta_notj).sum()
    tau = est * h - (h - 1) * theta_notj
    return np.sqrt(np.mean((tau - theta_J) ** 2 / (h - 1)))


# residualize the first num_background annotations out of the num_background+k-th
#   marginal annotation
# q is the numerator of the regression after background annots are residualized out
# r is the denominator of the regression after background annots are residualized out
# mux is the vector of coefficients required to residualize the background out of the
#   marginal annotatioin question
# muy is the vector of coefficients required to residualize the backgroud out of the
#   vector of GWAS summary statistics
def residualize(
    chunk_nums: list[np.ndarray],
    chunk_denoms: list[np.ndarray],
    num_background: int,
    k: int,
    total_num: np.ndarray | None = None,
    total_denom: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Residualize background annotations out of a marginal annotation."""

    q = np.array([num[num_background + k] for num in chunk_nums])
    r = np.array([denom[num_background + k, num_background + k] for denom in chunk_denoms])
    mux = np.array([])
    muy = np.array([])

    if num_background > 0:
        num = _sum_arrays(chunk_nums) if total_num is None else total_num
        denom = _sum_arrays(chunk_denoms) if total_denom is None else total_denom
        ATA = denom[:num_background][:, :num_background]
        ATy = num[:num_background]
        ATx = denom[:num_background][:, num_background + k]
        muy = np.linalg.solve(ATA, ATy)
        mux = np.linalg.solve(ATA, ATx)
        xiaiT = np.array([denom_chunk[num_background + k, :num_background] for denom_chunk in chunk_denoms])
        yiaiT = np.array([num_chunk[:num_background] for num_chunk in chunk_nums])
        aiaiT = np.array([denom_chunk[:num_background][:, :num_background] for denom_chunk in chunk_denoms])
        q = q - xiaiT.dot(muy) - yiaiT.dot(mux) + aiaiT.dot(muy).dot(mux)
        r = r - 2 * xiaiT.dot(mux) + aiaiT.dot(mux).dot(mux)

    return q, r, mux, muy


# do sign-flipping to get p-value
def signflip(
    q: np.ndarray,
    T: int,
    printmem: bool = True,
    mode: str = "sum",
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> tuple[float, float] | None:
    """Estimate a p-value by randomly sign-flipping independent chunk scores."""

    def mask(a, t):
        a_ = a.copy()
        a_[np.abs(a_) < t] = 0
        return a_

    if printmem:
        print("before sign-flipping:", fs.mem(), "MB")

    if mode == "sum":  # use sum of q as the test statistic
        score = q.sum()
    elif mode == "medrank":  # examine how far the rank of 0 deviates from the 50th percentile
        score = np.searchsorted(np.sort(q), 0) / len(q) - 0.5
    elif mode == "thresh":  # threshold q at some absolute magnitude threshold
        top = np.percentile(np.abs(q), 75)
        print(top)
        ts = np.arange(0, top, top / 10)
        q_thresh = np.array([mask(q, t) for t in ts]).T
        q_thresh /= np.linalg.norm(q_thresh, axis=0)
        scores = np.sum(q_thresh, axis=0)
        score = scores[np.argmax(np.abs(scores))]
    else:
        print("ERROR: invalid mode")
        return None

    null = np.zeros(T)
    current = 0
    block = 100000
    if rng is None:
        rng = np.random.RandomState()
    while current < len(null):
        s = (-1) ** rng.binomial(1, 0.5, size=(block, len(q)))
        if mode == "sum":
            null[current : current + block] = s.dot(q)
        elif mode == "medrank":
            null_q = s[:, :] * q[None, :]
            null_q = np.sort(null_q, axis=1)
            null[current : current + block] = np.array([np.searchsorted(s, 0) / len(s) - 0.5 for s in null_q])
        elif mode == "thresh":
            null_q_thresh = s[:, :, None] * q_thresh[None, :, :]
            sums = np.sum(null_q_thresh, axis=1)
            null[current : current + block] = np.array([s[np.argmax(np.abs(s))] for s in sums])

        current += block
        p = max(1, ((np.abs(null) >= np.abs(score)).sum())) / float(current)
        del s
        gc.collect()
        if p >= 0.01:
            null = cast(np.ndarray, null[:current])
            break

    if p >= 1:
        se = np.inf
    else:
        se = np.abs(score) / np.sqrt(st.chi2.isf(p, 1))
    del null
    gc.collect()
    if printmem:
        print("after sign-flipping:", fs.mem(), "MB. p=", p)

    return p, score / se
