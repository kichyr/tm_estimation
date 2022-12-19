import models
import torch
from matplotlib import pyplot as plt
from IPython.display import clear_output
import solvers.solver


class TMSolver_MSEMethod(solvers.solver.TMSolver):
    def __init__(
        self,
        max_grad_dec:int = 200,
        show_plt:bool = False,
        lr:float = 15e-3
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
        opt = torch.optim.SGD(likelihood_opt.parameters(), lr=self.lr)
        self.sample = None
        history = []
        for i in range(self.max_grad_dec):
            clear_output(True)
            opt.zero_grad()
            out = likelihood_opt()
            out.backward()
            torch.nn.utils.clip_grad_norm_(likelihood_opt.parameters(), 10)
            opt.step()
            history.append(out.detach())
            if i == 120:
                self.sample = torch.pow(likelihood_opt.lambdas.detach(), 2)
            if self.show_plt:
                print("error=" + str(history[-1]))
                plt.plot(history)
                plt.show()

        return (torch.pow(likelihood_opt.lambdas.detach(), 2), self.sample)

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
        i = 0
        hist_len = len(self.net_model.Y)
        for y in self.net_model.Y:
            i += 1
            likelihood += 1000 * i / hist_len * torch.norm(y - self.A @ torch.pow(self.lambdas, 2))

        likelihood += torch.linalg.norm(self.lambdas, dim=0)

        return likelihood