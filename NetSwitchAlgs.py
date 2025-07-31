import numpy as np
from scipy.sparse.linalg import eigsh
from sympy.strategies.core import switch
import itertools
import random
import igraph as ig
from numba import jit


@jit(nopython=True)
def assortativity_coeff(A, deg, zagreb=False):
    '''Calculates the assortativity coefficient for a graph
    from its binary adjacncy matrix.
    Calculations based on [PHYSICAL REVIEW E 84, 047101 (2011)].'''
    m = np.sum(A) / 2.0
    all_i, all_j = np.where(np.triu(A))
    M2 = np.sum(deg[all_i] * deg[all_j]) / m
    di1 = (np.sum(deg[all_i] + deg[all_j]) / (m * 2.0)) ** 2
    di2 = np.sum(deg[all_i] ** 2 + deg[all_j] ** 2) / (m * 2.0)
    if not zagreb:
        return (M2 - di1) / (di2 - di1), -1 #if (not zagreb) else ((M2 - di1) / (di2 - di1), M2)
    else:
        return (M2 - di1) / (di2 - di1), M2

@jit(nopython=True)
def remove_value(arr, val):
    mask = (arr != val)
    return arr[mask]

@jit(nopython=True)
def count_rowpair_checkers_fast_upperswt(A, i, j, pos_only=True):
    '''Similar to self.count_rowpair_checkers_fast(i, j)
    but only counting the switchings with k, l > i, j
    i.e., upper trianlge checkerboards'''
    if j < i:
        i, j = j, i
    all_checkerboard_sides = j + 1 + np.nonzero(A[i, j + 1:] ^ A[j, j + 1:])[0]
    all_checkerboard_sides = remove_value(all_checkerboard_sides, j)
    all_rightsides_pos = np.nonzero(A[i, all_checkerboard_sides])[0]
    if all_rightsides_pos.size == 0:
        pos_count = int(0)
    else:
        cumsum_checkers = np.cumsum(np.diff(all_rightsides_pos) - 1)
        pos_count = int(all_rightsides_pos[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers))

    if not pos_only:
        all_rightsides_neg = np.nonzero(1 - A[i, all_checkerboard_sides])[0]
        if all_rightsides_neg.size == 0:
            neg_count = int(0)
        else:
            cumsum_checkers = np.cumsum(np.diff(all_rightsides_neg) - 1)
            neg_count = int(all_rightsides_neg[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers))
        return pos_count, neg_count
    else:
        return pos_count, -1

@jit(nopython=True)
def count_rowpair_checkers_fast(A, i, j, pos_only=True):
    ''' Alternative to self.count_rowpair_checkers(i, j)
    Here the main operation is vectorized with Numpy for speed.
    For a pair (i, j) of rows in adjacency matrix A (i<j),
    checks all columns of A and counts all NEGATIVE checkerboards
    with coordinate (i, j, k, l). '''

    all_checkerboard_sides = i + 1 + np.nonzero(A[i, i + 1:] ^ A[j, i + 1:])[0]
    all_checkerboard_sides = remove_value(all_checkerboard_sides, j)
    all_rightsides_pos = np.nonzero(A[i, all_checkerboard_sides])[0]
    if all_rightsides_pos.size == 0:
        pos_count = int(0)
    else:
        cumsum_checkers = np.cumsum(np.diff(all_rightsides_pos) - 1)
        pos_count = int(all_rightsides_pos[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers))

    if not pos_only:
        all_rightsides_neg = np.nonzero(1 - A[i, all_checkerboard_sides])[0]
        if all_rightsides_neg.size == 0:
            neg_count = int(0)
        else:
            cumsum_checkers = np.cumsum(np.diff(all_rightsides_neg) - 1)
            neg_count = int(all_rightsides_neg[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers))
        return pos_count, neg_count
    else:
        return pos_count, -1

@jit(nopython=True)
def update_count_mat(A, swt, pc, pc_rows, nc=None, nc_rows=None, pos_only=True, count_upper=True):
    ''' Given a checkerboard (i, j, k, l) is switched,
    updates the matrix N that holds counts of checkerboards in matrix A '''

    for ref_row in swt:
        for row in range(ref_row):
            checker_count = np.array(count_rowpair_checkers_fast(A, row, ref_row, pos_only))
            if not count_upper:
                checker_count[0] -= count_rowpair_checkers_fast_upperswt(A, row, ref_row, pos_only)[0]
                checker_count[1] -= count_rowpair_checkers_fast_upperswt(A, row, ref_row, pos_only)[1]
            if pos_only:
                pc[row, ref_row] = checker_count[0]
                pc_rows[row] = np.sum(pc[row, :])
            else:
                pc[row, ref_row], nc[row, ref_row] = checker_count
                pc_rows[row] = np.sum(pc[row, :])
                nc_rows[row] = np.sum(nc[row, :])
        for row in range(ref_row + 1, np.shape(A)[0]):
            checker_count = np.array(count_rowpair_checkers_fast(A, ref_row, row, pos_only))
            if not count_upper:
                checker_count[0] -= count_rowpair_checkers_fast_upperswt(A, ref_row, row, pos_only)[0]
                checker_count[1] -= count_rowpair_checkers_fast_upperswt(A, ref_row, row, pos_only)[1]
            if pos_only:
                pc[ref_row, row] = checker_count[0]
            else:
                pc[ref_row, row], nc[ref_row, row] = checker_count
        pc_rows[ref_row] = np.sum(pc[ref_row, :])
        if not pos_only:
            nc_rows[ref_row] = np.sum(nc[ref_row, :])
    if pos_only:
        return pc, pc_rows, np.array([[-1]]), np.array([-1])
    else:
        return pc, pc_rows, nc, nc_rows


class NetMat:

    def __init__(self, g):
        self.A = np.array(g.get_adjacency().data) #, dtype=np.int8
        self.n = np.shape(self.A)[0]
        self.deg = self.degree_seq()

    def degree_seq(self):
        '''Returns the degree sequence of a graph from its adjacency matrix.'''
        return np.sum(self.A, axis=1)

    def assortativity_coeff(self, zagreb=False):
        '''Calculates the assortativity coefficient for a graph
        from its binary adjacncy matrix.
        Calculations based on [PHYSICAL REVIEW E 84, 047101 (2011)].'''
        return assortativity_coeff(self.A, self.deg, zagreb=zagreb)

    def laplacian(self):
        return np.diag(self.deg) - self.A
    def normalized_laplacian(self):
        Dm05 = np.diag(1 / np.sqrt(self.deg))
        return np.matmul(np.matmul(Dm05, self.laplacian()), Dm05)
    def l2(self, normed=True):
        if normed:
            eig_vals = eigsh(self.normalized_laplacian(), k=2, which='SM', return_eigenvectors=False)
        else:
            eig_vals = eigsh(self.laplacian(), k=2, which='SM', return_eigenvectors=False)
        return eig_vals[0]
    def lev(self):
        eig_val = eigsh(self.A.astype(float), k=1, which='LM', return_eigenvectors=False)[0]
        return eig_val

    def get_edges(self):
        edge_coords = np.where(np.triu(self.A) == 1)
        return edge_coords#set(zip(edge_coords[0], edge_coords[1]))


class NetSwitch(NetMat):

    def __init__(self, ig_graph, pos_only=True):
        super().__init__(ig_graph)
        self.sort_adj()
        self.countonce = True
        self.pos_only = pos_only  # False for both positive and negative switching
        self.checkercount_matrix()  # Counts the checkerboards for all pairs of nodes
        self.swt_done = 0

    def sort_adj(self):
        sortIdx = np.argsort(-self.deg)
        self.A = self.A[sortIdx, :][:, sortIdx]
        self.deg = self.deg[sortIdx]

    def checkercount_matrix(self, count_upper=True):
        ''' Builds a matrix N, where element N[i,j] counts
        NEGATIVE checkerboards in row-pair (i, j) of the adjacency matrix A. '''

        self.pc = np.zeros((self.n, self.n), dtype=np.int64)
        if not self.pos_only:
            self.nc = np.zeros((self.n, self.n), dtype=np.int64)

        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                if self.pos_only:
                    self.pc[i, j] = count_rowpair_checkers_fast(self.A, i, j, self.pos_only)[0] - (0 if count_upper else count_rowpair_checkers_fast_upperswt(self.A, i, j, self.pos_only)[0])
                    self.pc_rows = np.sum(self.pc, axis=1)
                else:
                    self.pc[i, j], self.nc[i, j] = np.array(count_rowpair_checkers_fast(self.A, i, j, self.pos_only)) - np.array((0 if count_upper else count_rowpair_checkers_fast_upperswt(self.A, i, j, self.pos_only)))
                    self.pc_rows = np.sum(self.pc, axis=1)
                    self.nc_rows = np.sum(self.nc, axis=1)

    def update_count_mat(self, swt, count_upper=True):
        ''' Given a checkerboard (i, j, k, l) is switched,
        updates the matrix N that holds counts of checkerboards in matrix A '''
        if self.pos_only:
            self.pc, self.pc_rows, _, _ = update_count_mat(self.A, swt, self.pc, self.pc_rows, pos_only=self.pos_only, count_upper=count_upper)
        else:
            self.pc, self.pc_rows, self.nc, self.nc_rows = update_count_mat(self.A, swt, self.pc, self.pc_rows, self.nc, self.nc_rows, pos_only=self.pos_only, count_upper=count_upper)

    def update_B(self, swt):
        ''' Given a checkerboard (i, j, k, l) is switched,
        updates the array B that holds size of row-pair checkerboards in matrix A '''

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

    def total_checkers(self, pos=None, both=False):
        """Returns the total number of checkerboards left in the adjacency matrix"""
        if self.pos_only:
            return np.sum(self.pc_rows)
        elif both:
            return np.sum(self.pc_rows) + np.sum(self.nc_rows)
        else:
            if pos is None:
                raise 'To count the total number of checkerboards their direction needs to be specified!'
            return np.sum(self.pc_rows) if pos else np.sum(self.nc_rows)

    def switch(self, swt, update_counts=True, update_B=False):
        """Switches a selected checkrboard in matrix A and calls for an update in checkerboard count
        given the coordinates (i, j, k, l), the checkerboars is at (i, k), (i, l), (j, k), (j, l)
        and the mirrored coordinates (k, i), (l, i), (k, j), (l, j) in matrix A"""
        i, j, k, l = swt
        self.A[i, k], self.A[i, l], self.A[j, k], self.A[j, l] = 1 - self.A[i, k], 1 - self.A[i, l], 1 - self.A[
            j, k], 1 - self.A[j, l]
        self.A[k, i], self.A[l, i], self.A[k, j], self.A[l, j] = 1 - self.A[k, i], 1 - self.A[l, i], 1 - self.A[
            k, j], 1 - self.A[l, j]
        if update_counts:
            self.update_count_mat(swt)
        if update_B:
            self.update_B(swt)

    def find_random_checker(self, pos=True):
        ''' Using the count matrices, draws a checkerboard at random
        :param pos: True (False) to find a negative (positive) checkerboard configuration at random
        :return: A list of four different nodes i, j, k, l determining the switch '''
        if not pos and self.pos_only:
            raise Exception("The object is not initiated for Negative switching!!!")
        elif pos:
            count_mat = self.pc
            count_mat_row = self.pc_rows
            shifter = 0
        elif not pos:
            count_mat = self.nc
            count_mat_row = self.nc_rows
            shifter = 1
        # FIND ROW I
        swt_idx = np.random.randint(np.sum(count_mat_row)) + 1
        Nrow_Cumsum = count_mat_row.cumsum()
        rnd_i = np.argwhere(Nrow_Cumsum >= swt_idx)[0][0]
        swt_idx -= 0 if (rnd_i == 0) else Nrow_Cumsum[rnd_i - 1]

        # FIND ROW J
        iRow_Cumsum = np.cumsum(count_mat[rnd_i, rnd_i + 1 :])
        rnd_j = np.argwhere(iRow_Cumsum >= swt_idx)[0][0]
        swt_idx -= 0 if (rnd_j == 0) else iRow_Cumsum[rnd_j - 1]
        rnd_j += rnd_i + 1

        # FIND COLUMNS K & L
        all_checkerboard_sides = rnd_i + 1 + np.nonzero(self.A[rnd_i, rnd_i + 1:] ^ self.A[rnd_j, rnd_i + 1:])[0]
        all_checkerboard_sides = np.delete(all_checkerboard_sides, np.where(all_checkerboard_sides == rnd_j))
        all_rightsides = np.nonzero(self.A[rnd_i, all_checkerboard_sides] - shifter)[0]
        cumsum_checkers = np.cumsum(np.cumsum(np.insert(np.diff(all_rightsides) - 1, 0, all_rightsides[0])))
        rnd_l = np.argwhere(cumsum_checkers >= swt_idx)[0][0]
        swt_idx -= 0 if (rnd_l == 0) else cumsum_checkers[rnd_l - 1]
        rnd_l = all_checkerboard_sides[all_rightsides[rnd_l]]
        rnd_k = all_checkerboard_sides[np.nonzero(self.A[rnd_j, all_checkerboard_sides] - shifter)[0][swt_idx - 1]]

        return (rnd_i, rnd_j, rnd_k, rnd_l)

    def get_all_checkers(self, row_i, row_j):
        all_checkerboard_sides = row_i + 1 + np.nonzero(self.A[row_i, row_i + 1:] ^ self.A[row_j, row_i + 1:])[0]
        all_checkerboard_sides = np.delete(all_checkerboard_sides, np.where(all_checkerboard_sides == row_j))
        all_ls = all_checkerboard_sides[np.nonzero(self.A[row_i, all_checkerboard_sides])[0]]
        all_ks = all_checkerboard_sides[np.nonzero(self.A[row_j, all_checkerboard_sides])[0]]
        all_kls = [i for i in itertools.product(all_ks, all_ls) if i[0] < i[1]]
        return random.sample(all_kls, len(all_kls))

    def batch_switch(self, row_i, row_j):
        all_checkerboard_sides = row_i + 1 + np.nonzero(self.A[row_i, row_i + 1:] ^ self.A[row_j, row_i + 1:])[0]
        all_checkerboard_sides = np.delete(all_checkerboard_sides, np.where(all_checkerboard_sides == row_j))
        all_ls = all_checkerboard_sides[np.nonzero(self.A[row_i, all_checkerboard_sides])[0]][::-1]
        all_ks = all_checkerboard_sides[np.nonzero(self.A[row_j, all_checkerboard_sides])[0]]
        min_size = np.min([all_ks.size, all_ls.size])
        all_ls = all_ls[:min_size]
        all_ks = all_ks[:min_size]
        batch_idxs = np.where(all_ls-all_ks>0)[0]

        self.A[row_i, all_ks[batch_idxs]] = 1
        self.A[row_i, all_ls[batch_idxs]] = 0
        self.A[row_j, all_ks[batch_idxs]] = 0
        self.A[row_j, all_ls[batch_idxs]] = 1

        self.A[all_ks[batch_idxs], row_i] = 1
        self.A[all_ls[batch_idxs], row_i] = 0
        self.A[all_ks[batch_idxs], row_j] = 0
        self.A[all_ls[batch_idxs], row_j] = 1
        self.update_N(swt=np.concatenate(([row_i, row_j], all_ks[batch_idxs], all_ls[batch_idxs]), axis=None))


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
            if len(to_calc) == 0: #No row-pair with unknown (not calculated) largest checkerboard
                break

            diag_ij = to_calc[0]
            ci, cj = self.diag2coord(diag_ij)
            row_dist = cj - ci
            best_possible = row_dist * (self.n - ci - 2 - (0 if row_dist > 1 else 1))
            if best_area >= best_possible: #The largest already-found checkerboard is unbeatable
                break

            if self.N[ci, cj] == 0: #No checkerboards for this row-pair
                self.B[diag_ij] = -1
                continue

            k, l = self.largest_kl(ci, cj)
            swt_area = (cj - ci) * (l - k)
            self.B[diag_ij] = swt_area
            if swt_area > best_area:
                best_swt = tuple([ci, cj, k, l])
                best_area = swt_area
        #print(self.B, self.total_checkers())
        return best_swt

    def switch_A(self, alg='RAND', count=-1, pos=None):
        """Performs a number of switchings with a specified algorithm on the adjacency matrix
        The number of switchings to perform is input by the 'count' argument
        count=-1 results in continous switchings until no checkerboard is left
        alg='RAND': selects a switching checkerboard at random"""
        if (not self.pos_only) and (pos is None):
            raise 'Switch direction has to be specified!!!'
        swt_num = 0
        if count == -1:
            count = self.total_checkers(pos=pos)
        while count > 0 and self.total_checkers(pos=pos) > 0:
            match alg:
                case 'RAND':
                    swt = self.find_random_checker(pos=pos)
                case 'ORDR':
                    if self.swt_done == 0:
                        self.i, self.j = 0, self.n - 1
                    swt = self.next_ij_rowrow()
                case 'ORDD':
                    if self.swt_done == 0:
                        self.i, self.j = 0, self.n - 1
                    swt = self.next_ij_diag()
                case 'BLOC':
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


                case 'SWPC':
                    if self.swt_done == 0:
                        self.org_nl2 = self.l2(normed=True)
                    cswitch_found = False
                    while not cswitch_found:
                        if self.total_checkers() == 0:
                            break
                        possible_rowpairs = np.where(self.N>0)
                        rand_rowpair_idx = np.random.randint(possible_rowpairs[0].size)
                        randi, randj = possible_rowpairs[0][rand_rowpair_idx], possible_rowpairs[1][rand_rowpair_idx]
                        all_kls = self.get_all_checkers(randi, randj)
                        for curk, curl in all_kls:
                            swt = randi, randj, curk, curl
                            self.switch(swt, update_N=False)
                            new_nl2 = self.l2(normed=True)
                            if new_nl2 >= self.org_nl2:
                                self.update_N(swt)
                                cswitch_found = True
                                break
                            else:
                                self.switch(swt, update_N=False)
                        if not cswitch_found:
                            self.N[randi, randj] = 0
                            self.update_Nrow(randi)

                case 'BEST':
                    if self.swt_done == 0:
                        self.B = np.zeros(int(self.n * (self.n - 1) / 2))
                    swt = self.next_best_swt()
                case 'GRDY':
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

            #i, j, k, l = swt
            #print([[self.A[i, k], self.A[i, l]], [self.A[j, k], self.A[j, l]]])
            if not alg=='SWPC' and not alg=='BLOC':
                self.switch(swt, update_B=(True if alg=='BEST' else False))
            if  alg=='SWPC' and not cswitch_found:
                self.swt_done -= 1
            self.swt_done += 1
            swt_num += 1
            count -= 1

        if self.pos_only:
            return swt_num if self.total_checkers() == 0 else -1
        else:
            return swt_num

    def XBS(self, pos_p=0.5, count=1, force_update_N = False):
        if pos_p == 1.0 and self.swt_done==0:
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
                    swt = np.empty(4, dtype='int')
                    swt[0], swt[1] = link_indices[0][link1], link_indices[1][link1]
                    swt[2], swt[3] = link_indices[0][link2], link_indices[1][link2]
                    if len(set(swt)) == 4:
                        break
            if pos_p > np.random.rand():
                argSort = np.argsort(swt)
                if self.A[swt[argSort[0]], swt[argSort[1]]] == 0 and self.A[swt[argSort[2]], swt[argSort[3]]] == 0:
                    # Condition is met to perform the assortative switch
                    self.A[swt[0], swt[1]], self.A[swt[1], swt[0]] = 0, 0
                    self.A[swt[2], swt[3]], self.A[swt[3], swt[2]] = 0, 0
                    self.A[swt[argSort[0]], swt[argSort[1]]], self.A[swt[argSort[1]], swt[argSort[0]]] = 1, 1
                    self.A[swt[argSort[2]], swt[argSort[3]]], self.A[swt[argSort[3]], swt[argSort[2]]] = 1, 1
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
        '''Havel-Hakimi Algorithm solution for
        assembling a graph given a degree sequence.
        This function returns False if the degree sequence is not graphic.'''
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
                sorted_nodes = sorted(sorted_nodes, key=lambda x: (x[0], -x[1]), reverse=True)
                v1 = sorted_nodes[0]
                this_degree = v1[0]
        if replace_adj:
            self.A = HH_adj
            return True
        else:
            return HH_adj

    def count_rowpair_checkers(self, i, j):
        ''' For a pair (i, j) of rows in adjacency matrix A (i<j),
        checks all columns of A and counts all NEGATIVE checkerboards
        with coordinate (i, j, k, l).
        This is following the paper's implementation. '''

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


    # def count_rowpair_checkers_fast_upperswt(self, i, j):
    #     '''Similar to self.count_rowpair_checkers_fast(i, j)
    #     but only counting the switchings with k, l > i, j
    #     i.e., upper trianlge checkerboards'''
    #     if j < i:
    #         i, j = j, i
    #     all_checkerboard_sides = j + 1 + np.nonzero(self.A[i, j + 1:] ^ self.A[j, j + 1:])[0]
    #     all_checkerboard_sides = np.delete(all_checkerboard_sides, np.where(all_checkerboard_sides == j))
    #     all_rightsides_pos = np.nonzero(self.A[i, all_checkerboard_sides])[0]
    #     if all_rightsides_pos.size == 0:
    #         pos_count = int(0)
    #     else:
    #         cumsum_checkers = np.cumsum(np.diff(all_rightsides_pos) - 1)
    #         pos_count = int(all_rightsides_pos[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers))
    #
    #     if not self.pos_only:
    #         all_rightsides_neg = np.nonzero(self.A[i, all_checkerboard_sides])[0]
    #         if all_rightsides_neg.size == 0:
    #             neg_count = int(0)
    #         else:
    #             cumsum_checkers = np.cumsum(np.diff(all_rightsides_neg) - 1)
    #             neg_count = int(all_rightsides_neg[0] * (cumsum_checkers.size + 1) + np.sum(cumsum_checkers))
    #         return pos_count, neg_count
    #
    #     return pos_count


    # def update_count_mat(self, swt, count_upper=True):
    #     ''' Given a checkerboard (i, j, k, l) is switched,
    #     updates the matrix N that holds counts of checkerboards in matrix A '''
    #     if self.pos_only:
    #         self.pc, self.pc_rows = update_count_mat(self.A, swt, self.pc, self.pc_rows, pos_only=True, count_upper=count_upper)
    #     else:
    #         self.pc, self.pc_rows, self.nc, self.nc_rows = update_count_mat(self.A, swt, self.pc, self.pc_rows, self.nc, self.nc_rows, pos_only=True, count_upper=count_upper)
    #     for ref_row in swt:
    #         for row in range(ref_row):
    #             checker_count = np.array(self.count_rowpair_checkers_fast(row, ref_row)) - np.array((0 if count_upper else self.count_rowpair_checkers_fast_upperswt(row, ref_row)))
    #             if self.pos_only:
    #                 self.pc[row, ref_row] = checker_count
    #                 self.pc_rows[row] = np.sum(self.pc[row, :])
    #             else:
    #                 self.pc[row, ref_row], self.nc[row, ref_row] = checker_count
    #                 self.pc_rows[row] = np.sum(self.pc[row, :])
    #                 self.nc_rows[row] = np.sum(self.nc[row, :])
    #         for row in range(ref_row + 1, self.n):
    #             checker_count = np.array(self.count_rowpair_checkers_fast(ref_row, row)) - np.array((0 if count_upper else self.count_rowpair_checkers_fast_upperswt(ref_row, row)))
    #             if self.pos_only:
    #                 self.pc[ref_row, row] = checker_count
    #             else:
    #                 self.pc[ref_row, row], self.nc[ref_row, row] = checker_count
    #         self.pc_rows[ref_row] = np.sum(self.pc[ref_row, :])
    #         if not self.pos_only:
    #             self.nc_rows[ref_row] = np.sum(self.nc[ref_row, :])