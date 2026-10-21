import sys

import numpy as np
import pandas as pd
from scipy import stats


def psi(old, new, bins=10):
    edges = np.quantile(old, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    o = np.histogram(old, edges)[0] / len(old)
    n = np.histogram(new, edges)[0] / len(new)
    o = np.clip(o, 1e-4, None)
    n = np.clip(n, 1e-4, None)
    return np.sum((n - o) * np.log(n / o))


reference = pd.read_csv("reference.csv")
new = pd.read_csv("new.csv")

alpha = 0.05 / len(reference.columns)
drifted = []

for col in reference.columns:
    ks = stats.ks_2samp(reference[col], new[col])
    score = psi(reference[col], new[col])
    flag = ks.pvalue < alpha and score > 0.1
    print(f"{col:15s} KS p={ks.pvalue:.4g}  PSI={score:.3f}  {'DRIFT' if flag else 'ok'}")
    if flag:
        drifted.append(col)

if drifted:
    print("Drift detected in:", drifted)
    sys.exit(1)

print("No drift detected.")
