import torch
from dijkstar import Graph, find_path
import itertools

import numpy as np


class NetflowProblemModel:
    """ Holds in it graph connectivity and history of links weight"""
    graph = torch.Tensor()
    # matrix Y=AX
    A = torch.Tensor()
    ListAk:list[torch.Tensor] = list()
    # 
    __graph = Graph()

    def __init__(
        self,
        graph:torch.Tensor,    
    ):
        self.graph = graph
        self.A =  torch.zeros(
            graph.size(dim=0) ** 2,
            graph.size(dim=0) ** 2, dtype=torch.float32)
        self.__calculate_A()
        self.ListAk = self.__calculate_A_k_list()

    def __pair_to_OD_number(self, i:int, j:int) -> int:
        return j + i*self.graph.size(dim=1)

    def __calculate_A(self):
        # crafting graph in format of dijkstar.Graph
        for i in range(self.graph.size(dim=0)):
            for j in range(self.graph.size(dim=1)):
                if self.graph[i][j] != 0:
                    self.__graph.add_edge(i, j, 1)
        # calculating best pathes and crafting matrix A
        for i in range(self.graph.size(dim=0)):
            for j in range(self.graph.size(dim=1)):
                if i == j:
                    self.A[self.__pair_to_OD_number(
                        i,j)][self.__pair_to_OD_number(
                        i,j)] = 1
                path = find_path(self.__graph, i, j)
                for node_index in range(len(path.nodes) - 1):
                    # set that (i,j) OD pair have best path trought  path[node_index],path[node_index+1] edge
                    self.A[self.__pair_to_OD_number(
                            path.nodes[node_index],path.nodes[node_index+1])][self.__pair_to_OD_number(
                        i,j)] = 1

    def __calculate_A_k_list(self) -> list[torch.Tensor]:
        resultedList = list()

        for i, row in enumerate(self.A):
            # Iterate over rows - rows represent all links in a graph
            # value 1 in a row repsesents in which OD pair this link is used

            A_k_matrix = torch.Tensor()
            
            for j, el in enumerate(row):
                if(el != 0):
                    new_eq = torch.zeros(
                    len(row), dtype=torch.float32)
                    new_eq[j] = 1
                    A_k_matrix = torch.cat([A_k_matrix, torch.Tensor([new_eq.tolist()])], axis=0)
            
            resultedList.append(A_k_matrix)

        return resultedList

    

    def find_optimal_solution(self, N:int, P:float) -> list[torch.tensor]:
        numberOfAks = len(self.ListAk)
        allWs = list()
        resultedWs = list()

        print("Calculating Ws list...")
        for i in range(0, numberOfAks):
            new_eq = torch.zeros(
                numberOfAks, dtype=torch.float32)
            new_eq[i] = 1
            allWs.append(new_eq)

        for i in range(1, N+1):
            allCombinations = itertools.combinations(allWs, i)

            for combo in allCombinations:
                wCombo = torch.zeros(
                    numberOfAks, dtype=torch.float32)
                for w in combo:
                    wCombo += w
                resultedWs.append(wCombo)
        print("Ok!")

        print("Calculating Mfs list...")
        allMfs = list()
        for w in resultedWs:
            Mf = torch.tensordot(torch.transpose(self.A, 0, 1), self.A, 1)
            for j, elem in enumerate(w):
                if (elem != 0 and len(self.ListAk[j]) != 0):
                    Mf += torch.tensordot(torch.transpose(self.ListAk[j], 0, 1), self.ListAk[j], 1)
            allMfs.append(Mf)
        print("Ok!")
        
        print("Calculating Phis list...")
        result = list()
        for Mf in allMfs:
            mfPow = torch.pow(Mf, P)
            semiRes = torch.trace(mfPow)
            result.append(semiRes)
        print("Ok")

        print("Searching for maximum...")
        resultMax = 0
        for i in range(0, len(result)):
            if result[i] > resultMax:
                resultMax = result[i]

        answer = list()
        for i in range(0, len(result)):
            if result[i] == resultMax:
                answer.append(resultedWs[i])

        return answer

