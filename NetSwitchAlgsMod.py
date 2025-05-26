import numpy as np
from scipy.sparse.linalg import eigsh

# from sympy.strategies.core import switch
import itertools
import random

import igraph as ig
from matplotlib import colors


class NetSwitch:

    def __init__(self, G, base=None, sortAdj=True):
        self.A = np.array(G.get_adjacency().data, dtype=np.int8)
        self.n = np.shape(self.A)[0]
        self.deg = self.degree_seq()
        self.m = np.sum(self.deg) / 2
        if sortAdj:
            self.sort_adj()
        self.countonce = True
        self.checkercount_matrix()
        self.swt_done = 0

        self.lapMat = None
        self.lapMat_N = None
        self.modMat = None
        self.modMat_N = None

        # -------------------------------------------Modularity Aware Modification---------------------------------------

        self.mLimStep = 0
        self.swt_rejected = 0

        self.M = np.zeros((self.n, self.n))
        self.m = np.sum(self.deg) / 2
        self.Dsqrt = np.diag(
            [
                1.0 / np.sqrt(self.deg[i]) if self.deg[i] != 0 else 0
                for i in range(self.n)
            ]
        )
        for i in range(self.n):
            for j in range(self.n):
                self.M[i, j] = self.A[i, j] - (self.deg[i] * self.deg[j]) / (2 * self.m)

        if not isinstance(base, list) and not isinstance(base, np.ndarray):
            self.base = np.zeros((self.n, self.n + 1))
            for u in range(self.n + 1):
                self.base[:, u] = np.array([-1 if i < u else 1 for i in range(self.n)])
                self.base[:, u] = self.base[:, u] / np.linalg.norm(self.base[:, u])
        else:
            self.set_Base(base)

        self.numbase = self.base.shape[1]

        self.base_mod_N = np.zeros(self.numbase)
        for u in range(self.numbase):
            self.base_mod_N[u] = (
                self.base[:, u].T @ self.Dsqrt @ self.M @ self.Dsqrt @ self.base[:, u]
            )

        self.base_mod = np.zeros(self.numbase)
        for u in range(self.numbase):
            self.base_mod[u] = self.base[:, u].T @ self.M @ self.base[:, u]

        self.base_cut = np.zeros(self.numbase)
        for u in range(self.numbase):
            self.base_cut[u] = self.base[:, u].T @ self.laplacian() @ self.base[:, u]

        self.base_cut_N = np.zeros(self.numbase)
        for u in range(self.numbase):
            self.base_cut_N[u] = (
                self.base[:, u].T @ self.normalized_laplacian() @ self.base[:, u]
            )

        self.M_ub = np.zeros(self.n + 1)
        self.M_lb = np.zeros(self.n + 1)
        self.degCumSum = np.cumsum(self.deg)
        for u in range(self.n + 1):
            s = self.base[:, u].T @ self.deg / np.sqrt(self.m * 2)
            outCnt = 0
            for v in range(u):
                if self.deg[v] >= u:
                    outCnt += self.deg[v] - u + 1

            self.M_ub[u] = (self.m * 2 - 4 * outCnt) / (self.n) - s * s

            outCnt = 0
            for v in range(u):
                outCnt = min(self.degCumSum[u - 1], 2 * self.m - self.degCumSum[u - 1])
                # min(sum(self.deg[:u]), sum(self.deg[u:]))

            self.M_lb[u] = (self.m * 2 - 4 * outCnt) / (self.n) - s * s

    def sort_adj(self, ordr=None):
        if not isinstance(ordr, (list, np.ndarray)):
            ordr = -self.deg
        sortIdx = np.argsort(ordr)
        self.A = self.A[sortIdx, :][:, sortIdx]
        self.deg = self.deg[sortIdx]

    def checkercount_matrix(self, count_upper=True):
        """Builds a matrix N, where element N[i,j] counts
        NEGATIVE checkerboards in row-pair (i, j) of the adjacency matrix A."""

        self.N = np.zeros((self.n, self.n), dtype=np.int64)
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                self.N[i, j] = self.count_rowpair_checkers_fast(i, j) - (
                    0
                    if count_upper
                    else self.count_rowpair_checkers_fast_upperswt(i, j)
                )
        self.Nrow = np.sum(self.N, axis=1)

    def count_rowpair_checkers(self, i, j):
        """For a pair (i, j) of rows in adjacency matrix A (i<j),
        checks all columns of A and counts all NEGATIVE checkerboards
        with coordinate (i, j, k, l).
        This is following the paper's implementation."""

        r, s = 0, 0
        k_init = 0 if self.countonce is False else i + 1
        for k in range(k_init, self.n):
            if k == i or k == j:
                continue
            if self.A[i, k] == 0 and self.A[j, k] == 1:
                r += 1
            if self.A[i, k] == 1 and self.A[j, k] == 0:
                s += r
        return s

    def count_rowpair_checkers_fast(self, i, j):
        """Alternative to self.count_rowpair_checkers(i, j)
        Here the main operation is vectorized with Numpy for speed.
        For a pair (i, j) of rows in adjacency matrix A (i<j),
        checks all columns of A and counts all NEGATIVE checkerboards
        with coordinate (i, j, k, l)."""

        all_checkerboard_sides = (
            i + 1 + np.nonzero(self.A[i, i + 1 :] ^ self.A[j, i + 1 :])[0]
        )
        all_checkerboard_sides = np.delete(
            all_checkerboard_sides, np.where(all_checkerboard_sides == j)
        )
        all_rightsides = np.nonzero(self.A[i, all_checkerboard_sides])[0]
        if all_rightsides.size == 0:
            return int(0)
        else:
            cumsum_checkers = np.cumsum(np.diff(all_rightsides) - 1)
            return int(
                all_rightsides[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers)
            )

    def count_rowpair_checkers_fast_upperswt(self, i, j):
        """Similar to self.count_rowpair_checkers_fast(i, j)
        but only counting the switchings with k, l > i, j
        i.e., upper trianlge checkerboards"""
        if j < i:
            i, j = j, i
        all_checkerboard_sides = (
            j + 1 + np.nonzero(self.A[i, j + 1 :] ^ self.A[j, j + 1 :])[0]
        )
        all_checkerboard_sides = np.delete(
            all_checkerboard_sides, np.where(all_checkerboard_sides == j)
        )
        all_rightsides = np.nonzero(self.A[i, all_checkerboard_sides])[0]
        if all_rightsides.size == 0:
            return int(0)
        else:
            cumsum_checkers = np.cumsum(np.diff(all_rightsides) - 1)
            return int(
                all_rightsides[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers)
            )

    def update_N(self, swt, count_upper=True):
        """Given a checkerboard (i, j, k, l) is switched,
        updates the matrix N that holds counts of checkerboards in matrix A"""

        for ref_row in swt:
            for row in range(ref_row):
                self.N[row, ref_row] = self.count_rowpair_checkers_fast(
                    row, ref_row
                ) - (
                    0
                    if count_upper
                    else self.count_rowpair_checkers_fast_upperswt(row, ref_row)
                )
                self.Nrow[row] = np.sum(self.N[row, :])
            for row in range(ref_row + 1, self.n):
                self.N[ref_row, row] = self.count_rowpair_checkers_fast(
                    ref_row, row
                ) - (
                    0
                    if count_upper
                    else self.count_rowpair_checkers_fast_upperswt(ref_row, row)
                )
                self.Nrow[ref_row] = np.sum(self.N[ref_row, :])

    def update_Nrow(self, rowi):
        self.Nrow[rowi] = np.sum(self.N[rowi, :])

    def update_B(self, swt):
        """Given a checkerboard (i, j, k, l) is switched,
        updates the array B that holds size of row-pair checkerboards in matrix A"""

        i, j, k, l = swt
        for ref_row in [i, j, k, l]:
            for row in range(ref_row):
                diag_idx = self.coord2diag(row, ref_row)
                if self.N[row, ref_row] == 0:
                    self.B[diag_idx] = -1
                elif self.B[diag_idx] != 0:
                    lft, rgt = self.largest_kl(row, ref_row)
                    self.B[diag_idx] = (ref_row - row) * (rgt - lft)
            for row in range(ref_row + 1, self.n):
                diag_idx = self.coord2diag(ref_row, row)
                if self.N[ref_row, row] == 0:
                    self.B[diag_idx] = -1
                elif self.B[diag_idx] != 0:
                    lft, rgt = self.largest_kl(ref_row, row)
                    self.B[diag_idx] = (row - ref_row) * (rgt - lft)

    def total_checkers(self):
        """Returns the total number of checkerboards left in the adjacency matrix"""
        return np.sum(self.Nrow)

    def switch(self, swt, update_N=True, update_B=False):
        """Switches a selected checkrboard in matrix A and calls for an update in checkerboard count
        given the coordinates (i, j, k, l), the checkerboars is at (i, k), (i, l), (j, k), (j, l)
        and the mirrored coordinates (k, i), (l, i), (k, j), (l, j) in matrix A"""
        i, j, k, l = swt
        self.A[i, k], self.A[i, l], self.A[j, k], self.A[j, l] = (
            1 - self.A[i, k],
            1 - self.A[i, l],
            1 - self.A[j, k],
            1 - self.A[j, l],
        )
        self.A[k, i], self.A[l, i], self.A[k, j], self.A[l, j] = (
            1 - self.A[k, i],
            1 - self.A[l, i],
            1 - self.A[k, j],
            1 - self.A[l, j],
        )
        # ---------------------------------Modularity Aware Modification--------------------------------
        self.update_M(swt)
        # ----------------------------------------------------------------------------------------------
        if update_N:
            self.update_N(swt)
        if update_B:
            self.update_B(swt)

    def find_random_checker(self, pos=True):
        if not pos:
            raise Exception(
                "Finding random negative checkerboards is not implemented yet!!!"
            )

        # FIND ROW I
        swt_idx = np.random.randint(np.sum(self.Nrow)) + 1
        Nrow_Cumsum = self.Nrow.cumsum()
        rnd_i = np.argwhere(Nrow_Cumsum >= swt_idx)[0][0]
        swt_idx -= 0 if (rnd_i == 0) else Nrow_Cumsum[rnd_i - 1]

        # FIND ROW J
        iRow_Cumsum = np.cumsum(self.N[rnd_i, rnd_i + 1 :])
        rnd_j = np.argwhere(iRow_Cumsum >= swt_idx)[0][0]
        swt_idx -= 0 if (rnd_j == 0) else iRow_Cumsum[rnd_j - 1]
        rnd_j += rnd_i + 1

        # FIND COLUMNS K & L
        all_checkerboard_sides = (
            rnd_i
            + 1
            + np.nonzero(self.A[rnd_i, rnd_i + 1 :] ^ self.A[rnd_j, rnd_i + 1 :])[0]
        )
        all_checkerboard_sides = np.delete(
            all_checkerboard_sides, np.where(all_checkerboard_sides == rnd_j)
        )
        all_rightsides = np.nonzero(self.A[rnd_i, all_checkerboard_sides])[0]
        cumsum_checkers = np.cumsum(
            np.cumsum(np.insert(np.diff(all_rightsides) - 1, 0, all_rightsides[0]))
        )
        rnd_l = np.argwhere(cumsum_checkers >= swt_idx)[0][0]
        swt_idx -= 0 if (rnd_l == 0) else cumsum_checkers[rnd_l - 1]
        rnd_l = all_checkerboard_sides[all_rightsides[rnd_l]]
        rnd_k = all_checkerboard_sides[
            np.nonzero(self.A[rnd_j, all_checkerboard_sides])[0][swt_idx - 1]
        ]

        return (rnd_i, rnd_j, rnd_k, rnd_l)

    def get_all_checkers(self, row_i, row_j):
        all_checkerboard_sides = (
            row_i
            + 1
            + np.nonzero(self.A[row_i, row_i + 1 :] ^ self.A[row_j, row_i + 1 :])[0]
        )
        all_checkerboard_sides = np.delete(
            all_checkerboard_sides, np.where(all_checkerboard_sides == row_j)
        )
        all_ls = all_checkerboard_sides[
            np.nonzero(self.A[row_i, all_checkerboard_sides])[0]
        ]
        all_ks = all_checkerboard_sides[
            np.nonzero(self.A[row_j, all_checkerboard_sides])[0]
        ]
        all_kls = [i for i in itertools.product(all_ks, all_ls) if i[0] < i[1]]
        return random.sample(all_kls, len(all_kls))

    def batch_switch(self, row_i, row_j):
        all_checkerboard_sides = (
            row_i
            + 1
            + np.nonzero(self.A[row_i, row_i + 1 :] ^ self.A[row_j, row_i + 1 :])[0]
        )
        all_checkerboard_sides = np.delete(
            all_checkerboard_sides, np.where(all_checkerboard_sides == row_j)
        )
        all_ls = all_checkerboard_sides[
            np.nonzero(self.A[row_i, all_checkerboard_sides])[0]
        ][::-1]
        all_ks = all_checkerboard_sides[
            np.nonzero(self.A[row_j, all_checkerboard_sides])[0]
        ]
        min_size = np.min([all_ks.size, all_ls.size])
        all_ls = all_ls[:min_size]
        all_ks = all_ks[:min_size]
        batch_idxs = np.where(all_ls - all_ks > 0)[0]

        self.A[row_i, all_ks[batch_idxs]] = 1
        self.A[row_i, all_ls[batch_idxs]] = 0
        self.A[row_j, all_ks[batch_idxs]] = 0
        self.A[row_j, all_ls[batch_idxs]] = 1

        self.A[all_ks[batch_idxs], row_i] = 1
        self.A[all_ls[batch_idxs], row_i] = 0
        self.A[all_ks[batch_idxs], row_j] = 0
        self.A[all_ls[batch_idxs], row_j] = 1
        self.update_N(
            swt=np.concatenate(
                ([row_i, row_j], all_ks[batch_idxs], all_ls[batch_idxs]), axis=None
            )
        )

    def largest_kl(self, row_i, row_j):
        for left_k in range(row_i + 1, self.n - 1):
            if left_k == row_j:
                continue
            if self.A[row_i, left_k] == 0 and self.A[row_j, left_k] == 1:
                break
        for rght_l in range(self.n - 1, left_k, -1):
            if rght_l == row_j:
                continue
            if self.A[row_i, rght_l] == 1 and self.A[row_j, rght_l] == 0:
                break
        return left_k, rght_l

    def largest_ij(self, col_k, col_l):
        for top_i in range(0, self.n - 3):
            if top_i == col_k or top_i == col_l:
                continue
            if self.A[top_i, col_k] == 0 and self.A[top_i, col_l] == 1:
                break
        for bot_j in range(self.n - 1, top_i, -1):
            if bot_j == col_k or bot_j == col_l:
                continue
            if self.A[bot_j, col_k] == 1 and self.A[bot_j, col_l] == 0:
                break
        return top_i, bot_j

    def next_ij_rowrow(self):
        while self.N[self.i, self.j] == 0:
            self.j -= 1
            if self.j == self.i:
                self.i += 1
                self.j = self.n - 1
                if self.i == self.n - 1:
                    self.i = 0
        ord_k, ord_l = self.largest_kl(self.i, self.j)
        return (self.i, self.j, ord_k, ord_l)

    def next_ij_diag(self):
        row_dist = self.j - self.i
        while self.N[self.i, self.j] == 0:
            self.i += 1
            self.j += 1
            if self.j == self.n:
                row_dist -= 1
                self.i = 0
                self.j = self.i + row_dist
                if row_dist == 0:
                    row_dist, self.i, self.j = self.n - 1, 0, self.n - 1
        ord_k, ord_l = self.largest_kl(self.i, self.j)
        return (self.i, self.j, ord_k, ord_l)

    def coord2diag(self, row, col):
        csarr = np.cumsum(np.arange(self.n - 1))[::-1]
        return csarr[col - row - 1] + row

    def diag2coord(self, idx):
        csarr = np.cumsum(np.arange(1, self.n))
        diag_idx = np.where(csarr - idx > 0)[0][0]
        i, j = diag_idx, self.n - 1
        i -= csarr[diag_idx] - 1 - idx
        j -= csarr[diag_idx] - 1 - idx
        return i, j

    def next_best_swt(self):
        best_area = np.max(self.B)
        bi, bj = self.diag2coord(np.argmax(self.B))
        bk, bl = self.largest_kl(bi, bj)
        best_swt = tuple([bi, bj, bk, bl])

        while True:
            to_calc = np.where(self.B == 0)[0]
            if (
                len(to_calc) == 0
            ):  # No row-pair with unknown (not calculated) largest checkerboard
                break

            diag_ij = to_calc[0]
            ci, cj = self.diag2coord(diag_ij)
            row_dist = cj - ci
            best_possible = row_dist * (self.n - ci - 2 - (0 if row_dist > 1 else 1))
            if (
                best_area >= best_possible
            ):  # The largest already-found checkerboard is unbeatable
                break

            if self.N[ci, cj] == 0:  # No checkerboards for this row-pair
                self.B[diag_ij] = -1
                continue

            k, l = self.largest_kl(ci, cj)
            swt_area = (cj - ci) * (l - k)
            self.B[diag_ij] = swt_area
            if swt_area > best_area:
                best_swt = tuple([ci, cj, k, l])
                best_area = swt_area
        # print(self.B, self.total_checkers())
        return best_swt

    def switch_A(self, alg="RAND", count=-1,**kwargs):
        """Performs a number of switchings with a specified algorithm on the adjacency matrix
        The number of switchings to perform is input by the 'count' argument
        count=-1 results in continous switchings until no checkerboard is left
        alg='RAND': selects a switching checkerboard at random"""
        swt_num = 0
        if count == -1:
            count = self.total_checkers()
        while count > 0 and self.total_checkers() > 0:
            match alg:
                case "RAND":
                    swt = self.find_random_checker()
                case "ORDR":
                    if self.swt_done == 0:
                        self.i, self.j = 0, self.n - 1
                    swt = self.next_ij_rowrow()
                case "ORDD":
                    if self.swt_done == 0:
                        self.i, self.j = 0, self.n - 1
                    swt = self.next_ij_diag()
                case "BLOC":
                    if self.swt_done == 0:
                        self.block_idx = 0
                    while True:
                        cur_i, cur_j = self.diag2coord(self.block_idx)
                        if self.N[cur_i, cur_j] == 0:
                            self.block_idx += 1
                            if self.block_idx == (self.n**2 - self.n) / 2:
                                self.block_idx = 0
                        else:
                            self.batch_switch(cur_i, cur_j)
                            break

                case "SWPC":
                    if self.swt_done == 0:
                        if "l2lim" not in kwargs:
                            self.org_nl2 = self.l2(normed=True)
                        else:
                            self.org_nl2 = self.l2(normed=True) * kwargs["l2lim"]

                    cswitch_found = False
                    while not cswitch_found:
                        if self.total_checkers() == 0:
                            break
                        possible_rowpairs = np.where(self.N > 0)
                        rand_rowpair_idx = np.random.randint(possible_rowpairs[0].size)
                        randi, randj = (
                            possible_rowpairs[0][rand_rowpair_idx],
                            possible_rowpairs[1][rand_rowpair_idx],
                        )
                        all_kls = self.get_all_checkers(randi, randj)
                        eig_vals, eig_vecs = eigsh(
                            self.normalized_laplacian(), k=2, which="SM"
                        )
                        fvec = eig_vecs[:, 1]
                        for curk, curl in all_kls:
                            swt = randi, randj, curk, curl
                            delta = (
                                fvec[randi] / np.sqrt(self.deg[randi])
                                - fvec[randj] / np.sqrt(self.deg[randj])
                            ) * (
                                fvec[curk] / np.sqrt(self.deg[curk])
                                - fvec[curl] / np.sqrt(self.deg[curl])
                            )
                            if delta > 0:
                                self.swt_rejected += 1
                                continue
                            self.switch(swt, update_N=False)
                            new_nl2 = self.l2(normed=True)
                            if new_nl2 >= self.org_nl2:
                                #print(self.swt_done)
                                self.update_N(swt)
                                cswitch_found = True
                                break
                            else:
                                self.swt_rejected += 1
                                self.switch(swt, update_N=False)
                        if not cswitch_found:
                            self.N[randi, randj] = 0
                            self.update_Nrow(randi)

                # ----------------------------------- Modularity Aware Modification -----------------------------------------#
                case "L2A-G":
                    if self.swt_done == 0:
                        self.org_nl2 = self.l2(normed=True)
                    cswitch_found = False
                    while not cswitch_found:
                        if self.total_checkers() == 0:
                            break
                        possible_rowpairs = np.where(self.N > 0)
                        rand_rowpair_idx = np.random.randint(possible_rowpairs[0].size)
                        randi, randj = (
                            possible_rowpairs[0][rand_rowpair_idx],
                            possible_rowpairs[1][rand_rowpair_idx],
                        )
                        all_kls = self.get_all_checkers(randi, randj)
                        eig_vals, eig_vecs = eigsh(
                            self.normalized_laplacian(), k=2, which="SM"
                        )
                        fvec = eig_vecs[:, 1]
                        for cnt, (curk, curl) in enumerate(all_kls):
                            swt0 = int(randi), int(randj), int(curk), int(curl)
                            if cnt < len(all_kls) * 0.01:
                                swts = self.expandSwitch(swt0)
                            # print("__________________\n", swts)
                            for swt in reversed(swts):
                                i, j, k, l = swt
                                # print(i, j, k, l, fvec[i], fvec[j], fvec[k], fvec[l])
                                delta = (
                                    fvec[i] / np.sqrt(self.deg[i])
                                    - fvec[j] / np.sqrt(self.deg[j])
                                ) * (
                                    fvec[k] / np.sqrt(self.deg[k])
                                    - fvec[l] / np.sqrt(self.deg[l])
                                )
                                if delta > 0:
                                    continue
                                self.switch(swt, update_N=False)
                                new_nl2 = self.l2(normed=True)
                                if new_nl2 >= self.org_nl2:
                                    print(self.swt_done)
                                    self.update_N(swt)
                                    cswitch_found = True
                                    break
                                else:
                                    self.swt_rejected += 1
                                    self.switch(swt, update_N=False)
                            if cswitch_found:
                                break
                        if not cswitch_found:
                            self.N[randi, randj] = 0
                            self.update_Nrow(randi)

                case "ModA-G":
                    if self.swt_done == 0:
                        self.m_limit = self.MScore(normed=False)
                    cswitch_found = False
                    while not cswitch_found:
                        if self.total_checkers() == 0:
                            break
                        possible_rowpairs = np.where(self.N > 0)
                        rand_rowpair_idx = np.random.randint(possible_rowpairs[0].size)
                        randi, randj = (
                            possible_rowpairs[0][rand_rowpair_idx],
                            possible_rowpairs[1][rand_rowpair_idx],
                        )
                        all_kls = self.get_all_checkers(randi, randj)
                        for cnt, (curk, curl) in enumerate(all_kls):
                            swt = randi, randj, curk, curl
                            if cnt < len(all_kls) * 0.05:
                                swt = self.expandSwitchModA(swt, normalized=False)
                            if self.checkOrdParMod(self.m_limit, swt, normalized=False):
                                # print(self.swt_done)
                                self.switch(swt, update_N=False)
                                self.update_N(swt)
                                cswitch_found = True
                                break
                            else:
                                self.swt_rejected += 1
                        if not cswitch_found:
                            self.N[randi, randj] = 0
                            self.update_Nrow(randi)

                case "ModA":
                    if self.swt_done == 0:
                        self.m_limit = self.MScore(normed=False)
                    cswitch_found = False
                    while not cswitch_found:
                        if self.total_checkers() == 0:
                            break
                        possible_rowpairs = np.where(self.N > 0)
                        rand_rowpair_idx = np.random.randint(possible_rowpairs[0].size)
                        randi, randj = (
                            possible_rowpairs[0][rand_rowpair_idx],
                            possible_rowpairs[1][rand_rowpair_idx],
                        )
                        all_kls = self.get_all_checkers(randi, randj)
                        for curk, curl in all_kls:
                            swt = randi, randj, curk, curl

                            if self.checkOrdParMod(self.m_limit, swt, normalized=False):
                                # print(self.swt_done)
                                self.switch(swt, update_N=False)
                                self.update_N(swt)
                                cswitch_found = True
                                break
                            else:
                                self.swt_rejected += 1
                        if not cswitch_found:
                            self.N[randi, randj] = 0
                            self.update_Nrow(randi)

                case "CutA":
                    if self.swt_done == 0:
                        # self.c_limit = max(self.base_cut)
                        self.c_limit = self.L2Score(normed=True)
                    cswitch_found = False
                    while not cswitch_found:
                        if self.total_checkers() == 0:
                            break
                        possible_rowpairs = np.where(self.N > 0)
                        rand_rowpair_idx = np.random.randint(possible_rowpairs[0].size)
                        randi, randj = (
                            possible_rowpairs[0][rand_rowpair_idx],
                            possible_rowpairs[1][rand_rowpair_idx],
                        )
                        all_kls = self.get_all_checkers(randi, randj)
                        for curk, curl in all_kls:
                            swt = randi, randj, curk, curl

                            if self.checkOrdParCut(self.c_limit, swt, normalized=True):
                                self.switch(swt, update_N=False)
                                self.update_N(swt)
                                cswitch_found = True
                                break
                            else:
                                self.swt_rejected += 1
                        if not cswitch_found:
                            self.N[randi, randj] = 0
                            self.update_Nrow(randi)

                # -----------------------------------------------------------------------------------------------------------#

                case "BEST":
                    if self.swt_done == 0:
                        self.B = np.zeros(int(self.n * (self.n - 1) / 2))
                    swt = self.next_best_swt()
                case "GRDY":
                    swt = self.find_random_checker()
                    i, j, k, l = swt
                    search_block = 0
                    while True:
                        new_k, new_l = self.largest_kl(i, j)
                        if new_k == k and new_l == l:
                            search_block += 1
                            if search_block == 2:
                                break
                        else:
                            k, l = new_k, new_l
                            search_block = 0
                        new_i, new_j = self.largest_ij(k, l)
                        if new_i == i and new_j == j:
                            search_block += 1
                            if search_block == 2:
                                break
                        else:
                            i, j = new_i, new_j
                            search_block = 0
                    # print(swt, i, j, k, l)
                    swt = i, j, k, l
                case _:
                    raise Exception("Undefined switching algorithm!!!")

            # i, j, k, l = swt
            # print([[self.A[i, k], self.A[i, l]], [self.A[j, k], self.A[j, l]]])=
            alg_tae = ["SWPC", "BLOC", "CutA", "ModA", "ModA-G", "L2A-G"]
            if alg not in alg_tae:
                self.switch(swt, update_B=(True if alg == "BEST" else False))
            if (alg in alg_tae) and not cswitch_found:
                self.swt_done -= 1
            self.swt_done += 1
            # print(self.swt_done)
            swt_num += 1
            count -= 1

        return swt_num if self.total_checkers() == 0 else -1

    def XBS(self, pos_p=0.5, count=1, force_update_N=False):
        if pos_p == 1.0 and self.swt_done == 0:
            self.checkercount_matrix(count_upper=False)
        swt_num = 0
        while count > 0 and (self.total_checkers() > 0 or pos_p < 1.0):
            link_indices = np.where(self.A == 1)
            while True:
                if pos_p == 1.0:
                    swt = self.find_random_checker()
                    swt = [swt[0], swt[3], swt[1], swt[2]]
                    break
                else:
                    link1, link2 = np.random.randint(len(link_indices[0]), size=2)
                    swt = np.empty(4, dtype="int")
                    swt[0], swt[1] = link_indices[0][link1], link_indices[1][link1]
                    swt[2], swt[3] = link_indices[0][link2], link_indices[1][link2]
                    if len(set(swt)) == 4:
                        break
            if pos_p > np.random.rand():
                argSort = np.argsort(swt)
                if (
                    self.A[swt[argSort[0]], swt[argSort[1]]] == 0
                    and self.A[swt[argSort[2]], swt[argSort[3]]] == 0
                ):
                    # Condition is met to perform the assortative switch
                    self.A[swt[0], swt[1]], self.A[swt[1], swt[0]] = 0, 0
                    self.A[swt[2], swt[3]], self.A[swt[3], swt[2]] = 0, 0
                    (
                        self.A[swt[argSort[0]], swt[argSort[1]]],
                        self.A[swt[argSort[1]], swt[argSort[0]]],
                    ) = (1, 1)
                    (
                        self.A[swt[argSort[2]], swt[argSort[3]]],
                        self.A[swt[argSort[3]], swt[argSort[2]]],
                    ) = (1, 1)
                    count -= 1
                    self.swt_done += 1
                    if pos_p == 1.0 or force_update_N:
                        self.update_N(swt, count_upper=False)
            elif self.A[swt[0], swt[3]] == 0 and self.A[swt[1], swt[2]] == 0:
                # Condition is met to perform the random switch
                self.A[swt[0], swt[1]], self.A[swt[1], swt[0]] = 0, 0
                self.A[swt[2], swt[3]], self.A[swt[3], swt[2]] = 0, 0
                self.A[swt[0], swt[3]], self.A[swt[3], swt[0]] = 1, 1
                self.A[swt[1], swt[2]], self.A[swt[2], swt[1]] = 1, 1
                count -= 1
                swt_num += 1
                self.swt_done += 1
                if pos_p == 1.0 or force_update_N:
                    self.update_N(swt, count_upper=False)
        return swt_num if (pos_p == 1.0 and self.total_checkers() == 0) else -1

    def Havel_Hakimi(self, replace_adj=False):
        """Havel-Hakimi Algorithm solution for
        assembling a graph given a degree sequence.
        This function returns False if the degree sequence is not graphic."""
        HH_adj = np.zeros((self.n, self.n))
        sorted_nodes = [i for i in zip(self.deg.copy(), range(self.n))]
        v1 = sorted_nodes[0]
        this_degree = v1[0]
        while this_degree > 0:
            if this_degree >= self.n:
                return False
            else:
                # Connecting the node with most remaining stubs to those sorted immediately after
                for v2_idx in range(1, this_degree + 1):
                    v2 = sorted_nodes[v2_idx]
                    # If condition met, the sequence is not graphic
                    if v2[0] == 0:
                        return False
                    else:
                        sorted_nodes[v2_idx] = (v2[0] - 1, v2[1])
                        HH_adj[v1[1], v2[1]], HH_adj[v2[1], v1[1]] = 1, 1
                sorted_nodes[0] = (0, sorted_nodes[0][1])
                # Re-sorting the nodes based on the count of remaining stubs
                sorted_nodes = sorted(
                    sorted_nodes, key=lambda x: (x[0], -x[1]), reverse=True
                )
                v1 = sorted_nodes[0]
                this_degree = v1[0]
        if replace_adj:
            self.A = np.array(HH_adj, dtype=np.int8)
            return True
        else:
            return HH_adj

    def degree_seq(self):
        """Returns the degree sequence of a graph from its adjacency matrix."""
        return np.sum(self.A, axis=1)

    def assortativity_coeff(self):
        """Calculates the assortativity coefficient for a graph
        from its binary adjacncy matrix.
        Calculations based on [PHYSICAL REVIEW E 84, 047101 (2011)]."""
        m = np.sum(self.A) / 2.0
        all_i, all_j = np.where(np.triu(self.A))
        M2 = np.sum(self.deg[all_i] * self.deg[all_j]) / m
        di1 = (np.sum(self.deg[all_i] + self.deg[all_j]) / (m * 2.0)) ** 2
        di2 = np.sum(self.deg[all_i] ** 2 + self.deg[all_j] ** 2) / (m * 2.0)
        return (M2 - di1) / (di2 - di1)

    def laplacian(self):
        self.lapMat = np.diag(self.deg) - self.A
        return self.lapMat

    def normalized_laplacian(self):
        Dm05 = np.diag(
            [1 / np.sqrt(self.deg[i]) if self.deg[i] != 0 else 0 for i in range(self.n)]
        )
        self.lapMat_N = np.matmul(np.matmul(Dm05, self.laplacian()), Dm05)

        return self.lapMat_N

    def normalized_modularity(self):
        if self.modMat_N == None:
            Dm05 = np.diag(
                [
                    1 / np.sqrt(self.deg[i]) if self.deg[i] != 0 else 0
                    for i in range(self.n)
                ]
            )
            self.modMat_N = Dm05 @ self.M @ Dm05

        return self.modMat_N

    def l2(self, normed=True, fast=True):
        if fast:
            if normed:
                eig_vals = eigsh(
                    self.normalized_laplacian(),
                    k=2,
                    which="SM",
                    return_eigenvectors=False,
                )
            else:
                eig_vals = eigsh(
                    self.laplacian(), k=2, which="SM", return_eigenvectors=False
                )
            eig_val = max(eig_vals)
        else:
            if normed:
                eig_val = np.linalg.eigvals(self.normalized_laplacian())
                idx = np.argsort(eig_val)
                # print(eig_val[idx[0:10]])
                eig_val = eig_val[idx[1]]
            else:
                eig_val = np.linalg.eigvals(self.laplacian())
                idx = np.argsort(eig_val)
                eig_val = eig_val[idx[1]]
        return eig_val

    def lev(self, fast=True):
        if fast:
            eig_val = eigsh(
                self.A.astype(float), k=1, which="LM", return_eigenvectors=False
            )[0]
        else:
            eig_val = np.linalg.eigvals(self.A.astype(float))
            idx = np.argsort(eig_val)
            eig_val = eig_val[idx[self.n - 1]]
        return eig_val

    def Mlev(self, normed=True, fast=True):
        if fast:
            if normed:
                eig_val = eigsh(
                    self.normalized_modularity().astype(float),
                    k=1,
                    which="LA",
                    return_eigenvectors=False,
                )[0]
            else:
                eig_val = eigsh(
                    self.M.astype(float), k=1, which="LA", return_eigenvectors=False
                )[0]
        else:
            if normed:
                eig_val = np.linalg.eigvals(self.normalized_modularity().astype(float))
                idx = np.argsort(eig_val)
                eig_val = eig_val[idx[self.n - 1]]
            else:
                eig_val = np.linalg.eigvals(self.M.astype(float))
                idx = np.argsort(eig_val)
                eig_val = eig_val[idx[self.n - 1]]
        return eig_val

    # -------------------------------------------Modularity Aware Modification---------------------------------------
    def MScore(self, normed=True, greedy=0):
        if normed:
            eig_val, eig_vec = np.linalg.eig(self.normalized_modularity().astype(float))
        else:
            eig_val, eig_vec = np.linalg.eig(self.M.astype(float))

        idx = np.argsort(eig_val)
        levec = eig_vec[:, idx[self.n - 1]]
        s = np.sign(levec.reshape(-1, 1)) / np.sqrt(self.n)

        if normed:
            score = (s.T @ self.normalized_modularity() @ s)[0, 0]
        else:
            score = (s.T @ self.M @ s)[0, 0]

        # print(levec)
        if greedy == 1:
            while True:
                mxidx = -1
                mxdelta = 0
                for i in range(self.n):
                    delta = 0
                    for j in range(self.n):
                        if i != j:
                            delta += -4 * s[i, 0] * self.M[i, j] * s[j, 0]
                    if delta > mxdelta:
                        mxidx = i
                        mxdelta = delta
                if mxdelta == 0:
                    break
                else:
                    s[mxidx, 0] = -s[mxidx, 0]
                    score = score + mxdelta
                print(score)
        return score
        if normed:
            return (eig_vec.T @ self.normalized_modularity() @ eig_vec)[0, 0]
        else:
            return (eig_vec.T @ self.M @ eig_vec)[0, 0]

    def L2Score(self, normed=True):
        if normed:
            eig_val, eig_vec = np.linalg.eig(self.normalized_laplacian().astype(float))
        else:
            eig_val, eig_vec = np.linalg.eig(self.laplacian().astype(float))
        idx = np.argsort(eig_val)
        eig_vec = np.sign(eig_vec[:, idx[1]].reshape(-1, 1)) / np.sqrt(self.n)
        if normed:
            return (eig_vec.T @ self.normalized_laplacian() @ eig_vec)[0, 0]
        else:
            return (eig_vec.T @ self.laplacian() @ eig_vec)[0, 0]

    def set_Base(self, base):
        self.base = base
        self.numbase = self.base.shape[1]
        for u in range(self.n):
            self.base[:, u] = self.base[:, u] / np.linalg.norm(self.base[:, u])

        self.base_mod_N = np.zeros(self.numbase)
        for u in range(self.numbase):
            self.base_mod_N[u] = (
                self.base[:, u].T @ self.Dsqrt @ self.M @ self.Dsqrt @ self.base[:, u]
            )

        self.base_mod = np.zeros(self.numbase)
        for u in range(self.numbase):
            self.base_mod[u] = self.base[:, u].T @ self.M @ self.base[:, u]

    def update_M(self, swt):
        i, j, k, l = swt

        self.M[i, k], self.M[i, l], self.M[j, k], self.M[j, l] = (
            self.M[i, k] - (1 - 2 * self.A[i, k]),
            self.M[i, l] - (1 - 2 * self.A[i, l]),
            self.M[j, k] - (1 - 2 * self.A[j, k]),
            self.M[j, l] - (1 - 2 * self.A[j, l]),
        )
        self.M[k, i], self.M[l, i], self.M[k, j], self.M[l, j] = (
            self.M[k, i] - (1 - 2 * self.A[k, i]),
            self.M[l, i] - (1 - 2 * self.A[l, i]),
            self.M[k, j] - (1 - 2 * self.A[k, j]),
            self.M[l, j] - (1 - 2 * self.A[l, j]),
        )

        for u in range(self.numbase):
            self.base_mod_N[u] = (
                self.base_mod_N[u]
                + (
                    self.base[i, u] / np.sqrt(self.deg[i])
                    - self.base[j, u] / np.sqrt(self.deg[j])
                )
                * (
                    self.base[k, u] / np.sqrt(self.deg[k])
                    - self.base[l, u] / np.sqrt(self.deg[l])
                )
                * 2
            )

            self.base_mod[u] = (
                self.base_mod[u]
                + (self.base[i, u] - self.base[j, u])
                * (self.base[k, u] - self.base[l, u])
                * 2
            )

            self.base_cut_N[u] = (
                self.base_cut_N[u]
                - (
                    self.base[i, u] / np.sqrt(self.deg[i])
                    - self.base[j, u] / np.sqrt(self.deg[j])
                )
                * (
                    self.base[k, u] / np.sqrt(self.deg[k])
                    - self.base[l, u] / np.sqrt(self.deg[l])
                )
                * 2
            )

            self.base_cut[u] = (
                self.base_cut[u]
                - (self.base[i, u] - self.base[j, u])
                * (self.base[k, u] - self.base[l, u])
                * 2
            )

    def checkOrdParMod(self, modularity_limit, swt, normalized=True):
        i, j, k, l = swt
        if not normalized:
            for u in range(self.numbase):
                new_modularity = (
                    self.base_mod[u]
                    + (self.base[i, u] - self.base[j, u])
                    * (self.base[k, u] - self.base[l, u])
                    * 2
                )
                if (
                    (
                        new_modularity > self.base_mod[u]
                        and self.base_mod[u] >= modularity_limit
                    )
                    or (
                        self.base_mod[u] < modularity_limit
                        and new_modularity >= modularity_limit
                    )
                    or (
                        self.base_mod[u] < self.M_ub[u]
                        and new_modularity >= self.M_ub[u]
                    )
                    # m_ratio
                ):
                    return False
        else:
            for u in range(self.numbase):
                new_modularity_N = (
                    self.base_mod_N[u]
                    + (
                        self.base[i, u] / np.sqrt(self.deg[i])
                        - self.base[j, u] / np.sqrt(self.deg[j])
                    )
                    * (
                        self.base[k, u] / np.sqrt(self.deg[k])
                        - self.base[l, u] / np.sqrt(self.deg[l])
                    )
                    * 2
                )

                if (
                    new_modularity_N > self.base_mod_N[u]
                    and self.base_mod_N[u] >= modularity_limit
                ) or (
                    self.base_mod_N[u] < modularity_limit
                    and new_modularity_N >= modularity_limit
                ):
                    return False
        return True

    def checkOrdParCut(self, cut_limit, swt, normalized=True):
        i, j, k, l = swt
        for u in range(self.numbase):
            new_cut = (
                self.base_cut[u]
                - (self.base[i, u] - self.base[j, u])
                * (self.base[k, u] - self.base[l, u])
                * 2
            )
            new_cut_N = (
                self.base_cut_N[u]
                - (
                    self.base[i, u] / np.sqrt(self.deg[i])
                    - self.base[j, u] / np.sqrt(self.deg[j])
                )
                * (
                    self.base[k, u] / np.sqrt(self.deg[k])
                    - self.base[l, u] / np.sqrt(self.deg[l])
                )
                * 2
            )
            if (
                not normalized
                and (new_cut < self.base_cut[u] and self.base_cut[u] <= cut_limit)
                or (self.base_cut[u] > cut_limit and new_cut <= cut_limit)
            ):
                return False
            elif (
                normalized
                and (new_cut_N < self.base_cut_N[u] and self.base_cut_N[u] <= cut_limit)
                or (self.base_cut_N[u] > cut_limit and new_cut_N <= cut_limit)
            ):
                return False

        return True

    def plotAdjacencyImage(self, ax, s=None):
        if not isinstance(s, list) and not isinstance(s, np.ndarray):
            _, s = np.linalg.eig(self.M.astype(float))
            idx = np.argsort(_)
            s = np.sign(np.real(s[:, idx[self.n - 1]])).reshape(-1, 1)

        score = (s.T @ self.M @ s)[0, 0]
        # print(score)
        # print(s)
        sPos = (s > 0).astype(np.float32).reshape(-1, 1)
        sNeg = (s < 0).astype(np.float32).reshape(-1, 1)
        img = (
            self.A
            + np.multiply(self.A, 2 * sPos @ sPos.T)
            - np.multiply(self.A, 2 * sNeg @ sNeg.T)
        )
        # sortIdx = np.argsort(s,stable = True)
        # img = img[sortIdx, :][:, sortIdx]
        _ = np.zeros((self.n, 1))
        _[0] = -1
        _[1] = 0
        _[2] = 1
        _[3] = 3
        cmap = colors.ListedColormap(["blue", "white", "green", "red"])
        img = np.hstack((img, _))
        ax.imshow(img, cmap=cmap)
        ax.set_xlim([-0.5, self.n - 0.5])
        ax.set_xticks([])
        ax.set_yticks([])

    def plotNetSwitchGraph(self, ax, s=[0], vertex_size=-1, edge_width=0.1):
        if s[0] == 0:
            _, s = np.linalg.eig(self.M.astype(float))
            idx = np.argsort(_)
            s = np.sign(np.real(s[:, idx[self.n - 1]]))

        if vertex_size == -1:
            vertex_size = 8000 / self.n
        # print(s)
        color = ["red" if i > 0 else "blue" for i in s]
        Gig = ig.Graph.Adjacency(self.A)
        edgecolor = [
            (0, 0.3, 0, 1) if s[i] != s[j] else (0, 0, 0, 0.05)
            for (i, j) in Gig.get_edgelist()
        ]
        edgewidth = [
            (
                np.log(self.n) * edge_width
                if s[i] != s[j]
                else np.log(self.n) * edge_width * 0.5
            )
            for (i, j) in Gig.get_edgelist()
        ]
        im3 = ig.plot(
            ig.Graph.Adjacency(self.A),
            vertex_size=np.log2(self.deg) * (vertex_size / np.log2(self.deg)[0]),
            edge_width=edgewidth,
            edge_arrow_size=0,
            edge_arrow_width=0,
            layout="circle",
            target=ax,
            vertex_color=color,
            edge_color=edgecolor,
            vertex_frame_width=0,
        )

    def expandSwitchModA(self, swt, normalized=True):
        i, j, k, l = swt
        search_block = 0
        bestswt = (i, j, k, l)

        while True:
            new_k, new_l = self.largest_kl(i, j)
            if new_k == k and new_l == l:
                search_block += 1
                if search_block == 2:
                    break
            else:
                k, l = new_k, new_l
                search_block = 0
            new_i, new_j = self.largest_ij(k, l)
            if new_i == i and new_j == j:
                search_block += 1
                if search_block == 2:
                    break
            else:
                i, j = new_i, new_j
                search_block = 0
            if self.checkOrdParMod(self.m_limit, (i, j, k, l), normalized=normalized):
                bestswt = (i, j, k, l)

        return bestswt

    def expandSwitch(self, swt):
        i, j, k, l = swt
        search_block = 0
        swts = [(i, j, k, l)]

        while True:
            new_k, new_l = self.largest_kl(i, j)
            if new_k == k and new_l == l:
                search_block += 1
                if search_block == 2:
                    break
            else:
                k, l = new_k, new_l
                search_block = 0
            new_i, new_j = self.largest_ij(k, l)
            if new_i == i and new_j == j:
                search_block += 1
                if search_block == 2:
                    break
            else:
                i, j = new_i, new_j
                search_block = 0
            swts.append((i, j, k, l))

        return swts

    def find_random_checker_mod(self):
        cnt = 0
        while cnt < 10000:
            cnt += 1
            rnd_i, rnd_j = random.sample(range(self.n), 2)

            all_checkerboard_sides = (
                rnd_i
                + 1
                + np.nonzero(self.A[rnd_i, rnd_i + 1 :] ^ self.A[rnd_j, rnd_i + 1 :])[0]
            )
            type1 = np.where(self.A[rnd_i, all_checkerboard_sides] == 0)[0]
            type2 = np.where(self.A[rnd_i, all_checkerboard_sides] == 1)[0]
            if len(type2) == 0 or len(type1) == 0:
                continue
            rnd_k, rnd_l = (
                all_checkerboard_sides[random.choice(type1)],
                all_checkerboard_sides[random.choice(type2)],
            )
            if len(set([rnd_i, rnd_j, rnd_k, rnd_l])) == 4:
                return (rnd_i, rnd_j, rnd_k, rnd_l)
        return (-1, -1, -1, -1)
