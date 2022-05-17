import torch
from dijkstar import Graph, find_path

# TM represents traffic matrix type
TM = torch.Tensor

class LinkWeight:
    """
    class LinkWeight represents weight of the link (i,j)
    """
    def __init__(self, i:int, j:int, weight:float):
        self.i = i
        self.j = j
        self.weight = weight

class NetworkModel:
    """ Holds in it graph connectivity and history of links weight"""
    graph = torch.Tensor()
    # matrix Y=AX
    A = torch.Tensor()
    # 
    __graph = Graph()

    def __init__(self, graph:torch.Tensor):
        self.graph = graph
        self.A =  torch.zeros(
            graph.size(dim=0) ** 2,
            graph.size(dim=0) ** 2, dtype=torch.float32)
        self.__calculate_A()

    def set_link_load_raw(self, Y):
        self.Y = Y

    def set_link_load(self, links_weights_hist):
        self.weights = []
        self.Y = []
        for h, link_weights in enumerate(links_weights_hist):
            self.Y.append(torch.zeros(self.graph.size(dim=0) ** 2, dtype=torch.int))
            self.weights.append(torch.zeros(size=self.graph.size()))
            for weight in link_weights:
                self.weights[h][weight.i, weight.j] = weight.weight

        self.calculate_Y()

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

    def calculate_Y(self):
        for w_index in range(len(self.weights)):
            for i in range(self.graph.size(dim=0)):
                for j in range(self.graph.size(dim=1)):
                    self.Y[w_index][self.__pair_to_OD_number(i,j)] = self.weights[w_index][i][j]
    
    def __pair_to_OD_number(self, i:int, j:int) -> int:
        return j + i*self.graph.size(dim=1)

