import numpy as np
from scipy.io import mmread

import scipy.sparse


def read_Graph(file, n=100, meanDeg=10):
    print(file)
    if file.lower().endswith((".edges")):
        with open(file, "r") as file:
            A = np.zeros((n, n))
            n = 1
            m = 0
            for line in file:
                words = line.rstrip().split()
                if words[0] == "%":
                    continue

                u, v = int(words[0]) - 1, int(words[1]) - 1
                n = max(n, u, v)
                if u >= A.shape[0] or v >= A.shape[0]:
                    pad_width = max(u, v) - A.shape[0] + 1
                    A = np.pad(
                        A,
                        ((0, pad_width), (0, pad_width)),
                        "constant",
                        constant_values=0,
                    )
                if u != v:
                    A[u, v] = 1
                    A[v, u] = 1
                    m += 1
            A = A[:n, :n]
            sortIdx = np.argsort(-np.sum(A, axis=0))
            A = A[sortIdx, :][:, sortIdx]
            return A
    elif file.lower().endswith((".mtx")):
        A = mmread(file)
        if scipy.sparse.issparse(A):
            A = np.array(A.todense())
        A = A.T + A
        A = np.array(A > 0, dtype=np.int8)
        sortIdx = np.argsort(-np.sum(A, axis=0))
        A = A[sortIdx, :][:, sortIdx]
        return A


# read_Graph("email-enron-only.mtx")
# A = read_Graph("ia-radoslaw-email.edges", n=200)

# print(np.sum(A, axis=0))
