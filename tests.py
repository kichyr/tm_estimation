import torch
import models
import likelihood_method
import numpy as np

def test_NetworkModel_init():
    graph = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    networkModel = models.NetworkModel(graph=graph)
    networkModel.set_link_load(
        links_weights_hist=[[models.LinkWeight(i=0,j=2,weight=10)]])
    assert(
        torch.equal(
            networkModel.weights[0],
            torch.Tensor([[0, 0, 10], [0, 0, 0], [0, 0, 0]])
        )
    )
    assert(
        torch.equal(
            networkModel.A,
            torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]],
            ).to(torch.float32),
        ),
    )
    print(networkModel.A)

def test_TMSolver_LikelihoodMethod():
    """
    [0]    [1]----10---->[2]
    """
    graph = torch.Tensor([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
    networkModel = models.NetworkModel(graph=graph)
    networkModel.set_link_load(
    links_weights_hist=[[models.LinkWeight(i=1,j=2,weight=10)]])

    likelihood_solver = likelihood_method.TMSolver_LikelihoodMethod()
    likelihood_solver.get_optimal_solution(networkModel)

    
