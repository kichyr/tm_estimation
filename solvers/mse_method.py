import models
import torch
from matplotlib import pyplot as plt
from IPython.display import clear_output
import solvers.solver


class TMSolver_MSEMethod(solvers.solver.TMSolver):
    def __init__(
        self,
        max_grad_dec:int = 200,
        show_plt:bool = False
    ):
        self.max_grad_dec = max_grad_dec
        self.show_plt = show_plt

    def get_optimal_solution(
        self,
        net_model:models.NetworkModel,
        ):
        """based on ..."""
        likelihood_opt = Optimizator(net_model)
        opt = torch.optim.SGD(likelihood_opt.parameters(), lr=15e-10)

        history = []
        for _ in range(self.max_grad_dec):
            opt.zero_grad()
            out = likelihood_opt()
            out.backward()
            torch.nn.utils.clip_grad_norm_(likelihood_opt.parameters(), 10)
            opt.step()
            history.append(out.detach())
            if self.show_plt:
                print("error=" + str(history[-1]))
                plt.plot(history)
                plt.show()
                clear_output(True)

        return (likelihood_opt.lambdas.detach(), 0)

class Optimizator(torch.nn.Module):
    def __init__(self, net_model:models.NetworkModel):
        torch.set_printoptions(profile="full")
        super().__init__()
        self.A = net_model.A
        # print(torch.det(self.A))
        self.net_model = net_model
        params_pattern = torch.ones(net_model.graph.size(dim=0)**2)
        self.lambdas = torch.nn.Parameter(params_pattern, requires_grad=True)

    def forward(self):
        likelihood = 0
        for y in self.net_model.Y:
            likelihood += torch.norm(y - self.A @ torch.pow(self.lambdas, 2))

        return likelihood