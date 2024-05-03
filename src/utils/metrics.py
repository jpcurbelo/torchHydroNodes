import numpy as np

## Evaluation metircs - after training
def NSE_eval(y_true, y_pred):
    
    numerator = np.sum(np.square(y_true - y_pred))
    denominator = np.sum(np.square(y_true - np.mean(y_true))) + np.finfo(float).eps

    nse_value = 1.0 - numerator / denominator

    return nse_value


if __name__ == "__main__":
    pass