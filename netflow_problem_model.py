import torch
from dijkstar import Graph, find_path


class NetflowProblemModel:
    """ Holds in it graph connectivity and history of links weight"""
    graph = torch.Tensor()
    # matrix Y=AX
    A = torch.Tensor()
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
                        i,j)][self.__pair_to_OD_number(
                            path.nodes[node_index],path.nodes[node_index+1])] = 1

    def __calculate_A_k_list(self) -> list[torch.Tensor]:
        pass