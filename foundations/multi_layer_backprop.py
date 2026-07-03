import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # convert to np.array
        x = np.asarray(x, dtype=float)
        W1 = np.asarray(W1, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        b2 = np.asarray(b2, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        # NO !
        # n is the nb of output features, not input
        # n = x.shape[0]

        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2

        n = z2.shape[0]

        error = z2 - y_true
        loss = np.mean((error) ** 2)

        dz2 = 2. * error / n              # (output)
        db2 = dz2
        dW2 = dz2[:, None] @ a1[None, :]    # (output, hidden)
        
        da1 = db2 @ W2
        relu_mask = z1 > 0
        dz1 = da1 * relu_mask
        db1 = dz1
        dW1 = dz1[:, None] @ x[None, :]


        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4),
            'db1': np.round(db1, 4),
            'dW2': np.round(dW2, 4),
            'db2': np.round(db2, 4)
        }

