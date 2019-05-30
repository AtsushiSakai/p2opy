"""

p2o 2D optimizer in Python

author: Atsushi Sakai

Ref:
    - [A Compact and Portable Implementation of Graph\-based SLAM](https://www.researchgate.net/publication/321287640_A_Compact_and_Portable_Implementation_of_Graph-based_SLAM)

    - [GitHub \- furo\-org/p2o: Single header 2D/3D graph\-based SLAM library](https://github.com/furo-org/p2o)

"""

import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt


class Optimizer2D:

    def __init__(self):
        self.p_lambda = 0.0
        self.verbose = False
        self.stop_thre = 1e-3
        self.robust_delta = sys.float_info.max

    def optimize_path(self, nodes, consts, max_iter, min_iter):

        graph_nodes = nodes[:]
        prev_cost = sys.float_info.max

        for i in range(max_iter):
            start = time.time()
            cost, graph_nodes = self.optimize_path_one_step(
                graph_nodes, consts)
            elapsed = time.time() - start
            if self.verbose:
                print("step ", i, " cost: ", cost, " time:", elapsed, "s")

            # check convergence
            if (i > min_iter) and (prev_cost - cost < self.stop_thre):
                if self.verbose:
                    print("converged:", prev_cost
                          - cost, " < ", self.stop_thre)
                    break

        return graph_nodes

    def optimize_path_one_step(self, graph_nodes, constraints):

        tripletList = []
        numnodes = len(graph_nodes)

        for con in constraints:
            ida = con.id1
            idb = con.id2
            assert 0 <= ida and ida < numnodes, "ida is invalid"
            assert 0 <= idb and idb < numnodes, "idb is invalid"
            pa = graph_nodes[ida]
            pb = graph_nodes[idb]

            r, Ja, Jb = self.calc_error(pa, pb, con.t)
            # info = con.info @ robust_coeff(r.transpose() * con.info * r, robust_delta);

        cost = 1.0

        return cost, graph_nodes

    def error_func(self, pa, pb, t):
        ba = self.ominus(pa, pb)
        error = np.array([ba.x - t.x,
                          ba.y - t.y,
                          self.normalize_rad_pi_mpi(ba.theta - t.theta)])
        return error

    def ominus(self, l, r):
        diff = np.array([l.x - r.x, l.y - r.y, l.theta - r.theta])
        v = np.matmul(self.rot_mat_2d(-r.theta), diff)
        v[2] = self.normalize_rad_pi_mpi(l.theta - r.theta)
        return Pose2D(v[0], v[1], v[2], None)

    def rot_mat_2d(self, th):
        return np.array([[math.cos(th), -math.sin(th), 0.0],
                         [math.sin(th), math.cos(th), 0.0],
                         [0.0, 0.0, 1.0]
                         ])

    def calc_error(self, pa, pb, t):

        e0 = self.error_func(pa, pb, t)
        dx = pb.x - pa.x
        dy = pb.y - pa.y
        dxdt = -math.sin(pa.theta) * dx + math.cos(pa.theta) * dy
        dydt = -math.cos(pa.theta) * dx - math.sin(pa.theta) * dy

        Ja = np.array([[-math.cos(pa.theta), -math.sin(pa.theta), dxdt],
                       [math.sin(pa.theta), -math.cos(pa.theta), dydt],
                       [0.0, 0.0, -1.0]
                       ])
        Jb = np.array([[math.cos(pa.theta), math.sin(pa.theta), 0.0],
                       [-math.sin(pa.theta), math.cos(pa.theta), 0.0],
                       [0.0, 0.0, 1.0]
                       ])

        return e0, Ja, Jb

    def normalize_rad_pi_mpi(self, rad):

        val = math.fmod(rad, 2.0 * math.pi)
        if val > math.pi:
            val -= 2.0 * math.pi
        elif val < -math.pi:
            val += 2.0 * math.pi

        return val


class Pose2D:

    def __init__(self, x, y, theta, data_id):
        self.x = x
        self.y = y
        self.theta = theta
        self.id = data_id


class Constrant2D:

    def __init__(self, id1, id2, t, info_mat):
        self.id1 = id1
        self.id2 = id2
        self.t = t
        self.info_mat = info_mat


def load_data(fname):
    nodes = []
    consts = []

    for line in open(fname):
        sline = line.split()
        tag = sline[0]

        if tag == "VERTEX_SE2":
            data_id = int(sline[1])
            x = float(sline[2])
            y = float(sline[3])
            theta = float(sline[4])
            nodes.append(Pose2D(x, y, theta, data_id))
        elif tag == "EDGE_SE2":
            id1 = int(sline[1])
            id2 = int(sline[2])
            x = float(sline[3])
            y = float(sline[4])
            th = float(sline[5])
            c1 = float(sline[6])
            c2 = float(sline[7])
            c3 = float(sline[8])
            c4 = float(sline[9])
            c5 = float(sline[10])
            c6 = float(sline[11])
            t = Pose2D(x, y, th, None)
            info_mat = np.array([[c1, c2, c3],
                                 [c2, c4, c5],
                                 [c3, c5, c6]
                                 ])
            consts.append(Constrant2D(id1, id2, t, info_mat))

    print("n_nodes:", len(nodes))
    print("n_consts:", len(consts))

    return nodes, consts


def main():
    print("start!!")

    fname = "./p2o/samples/intel.g2o"
    max_iter = 20
    min_iter = 3
    robust_thre = 1

    nodes, consts = load_data(fname)

    # parameter setting
    optimizer = Optimizer2D()
    optimizer.p_lambda = 1e-6
    optimizer.verbose = True
    optimizer.robust_thre = robust_thre

    start = time.time()
    optimizer.optimize_path(nodes, consts, max_iter, min_iter)
    print("elapsed_time", time.time() - start, "sec")

    # plotting
    for n in nodes:
        plt.plot(n.x, n.y, ".r", label="before")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    print("done!!")


if __name__ == '__main__':
    main()
