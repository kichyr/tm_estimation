import models
from abc import ABC, abstractmethod

class TMSolver(ABC):
    """ this class defines interface for TM solvers"""
    @abstractmethod
    def get_optimal_solution(self, net_model:models.NetworkModel):
        pass
