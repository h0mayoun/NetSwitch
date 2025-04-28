import numpy as np
from scipy.io import mmread, mmwrite

import scipy as sp
import sys

def tosparse(file):
    A = mmread(file)
    print(file)
    if sp.sparse.issparse(A):
            A = np.array(A.todense())
    A = A.T+A
    A = np.array(A > 0, dtype=np.int8)
    sortIdx = np.argsort(-np.sum(A, axis=0))
    A = A[sortIdx, :][:, sortIdx]
    mmwrite(file, sp.sparse.coo_matrix(A))
    
if __name__ == "__main__":
    tosparse(sys.argv[1])
    
    