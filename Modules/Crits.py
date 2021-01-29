import lstms.Modules.Functional as f
import numpy as np
class MSELoss:
    def loss(mean: bool, x: np.array, y: np.array):
        """
        :param mean: if the loss should be reduced to mean of all entries in resulting vector (if results in a vector)#bool
        :param x: input value/ prediction value#np.array
        :param y: target value#np.array
        :return: mean squared error#np.array
            :math:
                L = Sigma_n((x-y)^2))/n for n=number of entries in x,y; let dimensionality of x = y
            :math:
        """

        return ((x - y) ** 2) / len(x) if mean else (x - y) ** 2

