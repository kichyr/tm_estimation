import csv
from collections import deque
import models
import torch
import solvers.solver
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np

class BenchmarkUtil:
    networkModel:models.NetworkModel
    graph:torch.Tensor
    TM_hist:list()
    raw_TM_hist:list()

    def __init__(
        self,
        TM_hist:list=[],
        graph:torch.Tensor=torch.Tensor([]),  
    ):
        self.TM_hist = TM_hist
        self.graph = graph
        self.raw_TM_hist = deque(maxlen=len(self.TM_hist))
        self.networkModel = models.NetworkModel(graph=graph)
        for tm in self.TM_hist:
            self.raw_TM_hist.append(convert_2d_matrix_to_raw_tm(tm))

        Ys, _ = convert_to_link_weights(self.networkModel.A, self.raw_TM_hist)
        self.networkModel.set_link_load_raw(Ys)

    def load_default_TM_from_file(
        self,
        history_size = 10,
        started_sample_index = 0,    
    ) -> None:
        self.graph = test_graph
        self.networkModel = models.NetworkModel(graph=self.graph)
        self.TM_hist = []
        links_weights_hist = deque(maxlen=history_size)

        with open('data/geant-flat-tms.csv.1', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            i = 0
            for row in spamreader:
                i+=1
                rawTM = [float(x) for x in row[0].split(",")[1:]]
                # set follow from i to i node as zero
                for j in range(self.networkModel.graph.size(dim=0)):
                    rawTM[j * self.networkModel.graph.size(dim=0) + j] = 0
                links_weights_hist.append(rawTM)
                if i < history_size  or i < started_sample_index:
                    continue
                    
                Ys, _ = convert_to_link_weights(self.networkModel.A, links_weights_hist)
                self.networkModel.set_link_load_raw(Ys)

                self.TM_hist.append(convert_raw_tm_2d_matrix(raw_TM=rawTM, n=self.networkModel.graph.size(dim=0)))

    def activate_netflow_in_model(self, w:list):
        A_netflow, Y_hist = self.__generate_netflow_equations(w)
        self.networkModel.add_netflow_equations(
            A_netflow, Y_hist,
        )

    def __generate_netflow_equations(self, w:list):
        A_netflow = torch.empty(0,self.networkModel.A.size(dim=1))
        for i, row in enumerate(self.networkModel.A):
            if(w[i] == 0):
                continue
            for j, el in enumerate(row):
                if(el != 0):
                    new_eq = torch.zeros(
                    len(row), dtype=torch.float32)
                    new_eq[j] = 1
                    A_netflow = torch.cat([A_netflow, torch.Tensor([new_eq.tolist()])], axis=0)

        Y_hist = []
        for tm in self.raw_TM_hist:
            Y_hist.append(A_netflow @ tm)

        return A_netflow, Y_hist

    ### benchmarks graphs
    def show_benchmark_for_last_TM(
        self,
        solver:solvers.solver.TMSolver,
        debug=False,
    ):

        loss_historyX = []
        loss_historyY = []

        for rawTM in self.raw_TM_hist:
            predTM, sample = solver.get_optimal_solution(self.networkModel)
            (lossX, lossY) = loss(self.networkModel.A, predTM, torch.Tensor(rawTM), sample=sample, debug=debug)
            loss_historyX.append(lossX)
            loss_historyY.append(lossY)


        plt.figure(figsize=(12,12))
        plt.plot(loss_historyX, label="X error")
        plt.plot(loss_historyY, label="Y error")
        plt.xlabel("experiment number")
        plt.ylabel("relative loss")
        plt.legend()
        plt.show()



    ### utils methods

    def show_TM_heat_map(self):
        plt.imshow(
            convert_raw_tm_2d_matrix(
                raw_TM=self.TM_hist[0],
                n=self.graph.size(dim=0)),
            cmap='hot',
            interpolation='nearest',
        )
        plt.show()
    

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
        Xs.append(X)
        Ys.append(A @ X)
    return (Ys, Xs)

def loss(A, predTM, originalTM, sample=None, debug=False):
    diff = predTM - originalTM
    diffY = A @ predTM - A @ originalTM
    if debug:
        print(str(torch.norm(diff)))
        print(str(torch.norm(originalTM)))
    if sample is not None:
        return (torch.norm(A @ sample - A @ originalTM) / (torch.norm(A @ originalTM)), torch.norm(diffY) / torch.norm(A @ originalTM))
    return  (torch.norm(diff) / (torch.norm(originalTM)), torch.norm(diffY) / torch.norm(A @ originalTM))
    # return (float((torch.abs(diff.sum() / diff.size(dim=0))) / (torch.abs(originalTM.sum() / originalTM.size(dim=0)))),
    # float((torch.abs(diffY.sum() / diffY.size(dim=0))) / (torch.abs((A @ originalTM).sum() / (A @ originalTM).size(dim=0))))
    # )

def convert_raw_tm_2d_matrix(raw_TM, n):
    TM = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            TM[i][j] = raw_TM[i * n + j]
    return TM

def convert_2d_matrix_to_raw_tm(TM):
    raw_TM = np.zeros(len(TM) ** 2)
    for i in range(len(TM)):
        for j in range(len(TM)):
            raw_TM[i * len(TM) + j] = TM[i][j]
    return raw_TM


def benchmark_solver(
    solver:solvers.solver.TMSolver,
    test_cases_size = 1,
    history_size = 10,
    started_sample_index = 0,
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
            if i < history_size  or i < started_sample_index:
                continue
                
            Ys, _ = convert_to_link_weights(networkModel.A, links_weights_hist)
            networkModel.set_link_load_raw(Ys)

            plt.imshow(
                convert_raw_tm_2d_matrix(raw_TM=rawTM, n=networkModel.graph.size(dim=0)),
                cmap='hot',
                interpolation='nearest',
            )
            return

            predTM, sample = solver.get_optimal_solution(networkModel)
            (lossX, lossY) = loss(networkModel.A, predTM, torch.Tensor(rawTM), sample=sample, debug=debug)
            loss_historyX.append(lossX)
            loss_historyY.append(lossY)


            plt.figure(figsize=(12,12))
            plt.plot(loss_historyX, label="X error")
            plt.plot(loss_historyY, label="Y error")
            plt.xlabel("experiment number")
            plt.ylabel("relative loss")
            plt.legend()
            plt.show()
            clear_output(True)
            if i - history_size + 1 == started_sample_index + test_cases_size:
                return
            