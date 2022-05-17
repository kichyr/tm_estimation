import csv
from collections import deque
import models
import torch
import solvers.solver
from matplotlib import pyplot as plt
from IPython.display import clear_output

test_graph = torch.Tensor([
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 1
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], # 2
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 3
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 4
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 5
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 6
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0], # 7
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 8
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # 9
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 10
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], # 11
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # 12
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 13
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 14
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 15
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 16
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 17
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 18
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], # 20
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 21
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 22
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 23
])

def convert_to_link_weights(A, TMs):
    Ys = []
    Xs = []
    for tm in TMs:
        X = torch.Tensor(tm)
        X = X / X.max()
        Xs.append(X)
        Ys.append(A @ X)
    return (Ys, Xs)

def loss(A, predTM, originalTM, debug=False):
    diff = predTM - originalTM
    diffY = A @ predTM - A @ originalTM
    if debug:
        print(diff)
    
    return  (torch.norm(diff) / torch.norm(originalTM), torch.norm(diffY) / torch.norm(A @ originalTM))
    # return (float((torch.abs(diff.sum() / diff.size(dim=0))) / (torch.abs(originalTM.sum() / originalTM.size(dim=0)))),
    # float((torch.abs(diffY.sum() / diffY.size(dim=0))) / (torch.abs((A @ originalTM).sum() / (A @ originalTM).size(dim=0))))
    # )


def benchmark_solver(
    solver:solvers.solver.TMSolver,
    test_cases_size = 1,
    history_size = 10,
    debug=False,
):
    links_weights_hist = deque(maxlen=history_size)
    networkModel = models.NetworkModel(graph=test_graph)

    with open('data/geant-flat-tms.csv.1', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        i = 0
        loss_historyX = []
        loss_historyY = []
        for row in spamreader:
            i+=1
            rawTM = [float(x) for x in row[0].split(",")[1:]]
            # set follow from i to i node as zero
            for j in range(networkModel.graph.size(dim=0)):
                rawTM[j * networkModel.graph.size(dim=0) + j] = 0
            links_weights_hist.append(rawTM)
            if i < history_size:
                continue
                
            Ys, _ = convert_to_link_weights(networkModel.A, links_weights_hist)
            networkModel.set_link_load_raw(Ys)
            predTM, _ = solver.get_optimal_solution(networkModel)
            print(networkModel.A @ predTM)
            (lossX, lossY) = loss(networkModel.A, predTM, torch.Tensor(rawTM), debug=debug)
            loss_historyX.append(lossX)
            loss_historyY.append(lossY)

            plt.plot(loss_historyX)
            plt.plot(loss_historyY)
            plt.show()
            clear_output(True)
            if i - history_size + 1 == test_cases_size:
                return
            