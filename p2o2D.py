""" 

p2o 2D optimizer in Python

author: Atsushi Sakai

Ref:
    - [A Compact and Portable Implementation of Graph\-based SLAM](https://www.researchgate.net/publication/321287640_A_Compact_and_Portable_Implementation_of_Graph-based_SLAM)

    - [GitHub \- furo\-org/p2o: Single header 2D/3D graph\-based SLAM library](https://github.com/furo-org/p2o)

"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt


class Optimizer2D:

    def __init__(self):
        self.p_lambda = 0.0
        self.verbose = False
        self.stop_thre = 1e-3
        self.robust_delta = sys.float_info.max

    def optimize_path(self, nodes, consts, max_iter, min_iter):

        graph_nodes = []

        for i in range(max_iter):
            start = time.time()
            cost, graph_nodes = self.optimize_path_one_step()
            elapsed = time.time() - start
            if self.verbose:
                print("step ", i, ": ", cost, " time:", elapsed, "s")

        return graph_nodes

    def optimize_path_one_step(self):

        cost = 1.0
        graph_nodes = []

        return cost, graph_nodes


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
