import sympy
from sympy import Matrix, Symbol
from sympy.tensor.array import Array

def matrixform(data):
    """
    Translates an Array or nested list, or anything that behaves
    similarly, into a structure of nested matrices.
    """
    try:
        A = data.tolist()
    except AttributeError:
        A = data
        
    try:
        M = Matrix(A)
        return M.applyfunc(matrixform)
    except TypeError:
        return A
