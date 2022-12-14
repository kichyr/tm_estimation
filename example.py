import benchmark_util
import solvers.mse_method
import matplotlib.pyplot as plt


mse_solver = solvers.mse_method.TMSolver_MSEMethod(max_grad_dec = 50, show_plt=True)
benchmark_util.benchmark_solver(mse_solver, test_cases_size=1, history_size = 1, debug=False)
plt.show()