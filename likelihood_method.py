import models
import torch
from matplotlib import pyplot as plt
from IPython.display import clear_output


class TMSolver_LikelihoodMethod(models.TMSolver):
    def get_optimal_solution(
        self,
        net_model:models.NetworkModel,
        max_grad_dec:int = 200,
        show_plt:bool = False,
        ):
        """based on ..."""
        likelihood_opt = Optimizator(net_model)
        opt = torch.optim.SGD(likelihood_opt.parameters(), lr=3e-1)

        history = []
        for _ in range(max_grad_dec):
            opt.zero_grad()
            out = likelihood_opt()
            out.backward()
            opt.step()
            history.append(out.detach())
            if show_plt:
                plt.plot(history)
                plt.show()
                clear_output(True)

        return (likelihood_opt.lambdas, likelihood_opt.phi)

class Optimizator(torch.nn.Module):
    def __init__(self, net_model:models.NetworkModel):
        super().__init__()
        self.A = net_model.A
        self.net_model = net_model
        self.lambdas = torch.nn.Parameter(torch.ones(net_model.graph.size(dim=0)**2), requires_grad=True)
        self.phi = torch.nn.Parameter(torch.tensor(1,dtype=torch.float64), requires_grad=True)

    def forward(self):
        self.sigma = self.phi * torch.pow(torch.diag(self.lambdas), 2)
        likelihood = torch.log(torch.linalg.norm(self.A@self.sigma@self.A.T))
        for y in self.net_model.Y:
            likelihood -= 0.5 * (y - self.A @ self.lambdas).T @ torch.inverse(
                self.A@self.sigma@self.A.T) @ (y-self.A@self.lambdas)

        return -likelihood