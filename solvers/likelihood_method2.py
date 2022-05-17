import models
import torch
from matplotlib import pyplot as plt
from IPython.display import clear_output
import solvers.solver


class TMSolver_LikelihoodMethod(solvers.solver.TMSolver):
    def __init__(
        self,
        max_grad_dec:int = 200,
        show_plt:bool = False,
        lr:float = 15e-1
    ):
        self.max_grad_dec = max_grad_dec
        self.show_plt = show_plt
        self.lr = lr

    def get_optimal_solution(
        self,
        net_model:models.NetworkModel,
        ):
        """based on ..."""
        likelihood_opt = Optimizator(net_model)
        opt = torch.optim.Adam(likelihood_opt.parameters(), lr=self.lr)

        history = []
        for _ in range(self.max_grad_dec):
            opt.zero_grad()
            out = likelihood_opt()
            out.backward()
            for i in range(net_model.graph.size(dim=0)):
                likelihood_opt.lambdas.grad[i * net_model.graph.size(dim=0) + i] = 0
            opt.step()
            print(out.detach())
            history.append(out.detach())
            if self.show_plt:
                plt.plot(history)
                plt.show()
                clear_output(True)

        return (likelihood_opt.lambdas, likelihood_opt.phi)

class Optimizator(torch.nn.Module):
    def __init__(self, net_model:models.NetworkModel):
        super().__init__()
        self.A = net_model.A
        self.A = net_model.A + 0.001 * torch.eye(self.A.size(dim=0))
        self.net_model = net_model
        params_pattern = torch.ones(net_model.graph.size(dim=0)**2)
        for i in range(net_model.graph.size(dim=0)):
            params_pattern[i * net_model.graph.size(dim=0) + i] = 0.0001
        self.lambdas = torch.nn.Parameter(params_pattern, requires_grad=True)
        self.phi = torch.nn.Parameter(torch.tensor(1,dtype=torch.float64), requires_grad=True)

    def forward(self):
        self.sigma = self.phi * torch.pow(torch.diag(self.lambdas), 2)
        self.sigma = self.phi * torch.eye(self.A.size(dim=0))
        likelihood = - len(self.net_model.Y) / 2.0 * torch.log(torch.det(self.A@self.sigma@self.A.T))
        if torch.isnan(torch.det(self.A@self.sigma@self.A.T)):
             raise ValueError('A@sigma@A.T is nan det')
        for y in self.net_model.Y:
            likelihood -= 0.5 * (y - self.A @ self.lambdas).T @ torch.inverse(
                self.A@self.sigma@self.A.T) @ (y-self.A@self.lambdas)

        return -likelihood