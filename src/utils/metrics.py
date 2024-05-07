import numpy as np
from scipy import signal, stats

## Evaluation metircs - after training
def NSE_eval(y_true, y_pred):
    '''
    Nash-Sutcliffe Efficiency (NSE) metric
    NSE = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))
    
    -Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        
    -Returns:
        nse_value: float, NSE value
        
    References
    ----------
    .. [#] Nash, J. E.; Sutcliffe, J. V. (1970). "River flow forecasting through conceptual models part I - A 
        discussion of principles". Journal of Hydrology. 10 (3): 282-290. doi:10.1016/0022-1694(70)90255-6.
    '''
    
    numerator = np.sum(np.square(y_true - y_pred))
    denominator = np.sum(np.square(y_true - np.mean(y_true))) + np.finfo(float).eps

    nse_value = 1.0 - numerator / denominator

    return nse_value

def alphaNSE_eval(y_true, y_pred):
    '''
    Alpha-NSE metric
    
    -Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        
    -Returns:
        alpha_nse: float, Alpha-NSE decomposition
    
    References
    ----------
    .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error 
        and NSE performance criteria: Implications for improving hydrological modelling. Journal of hydrology, 377(1-2),
        80-91.
    '''
    
    return y_pred.std() / y_true.std()

def betaNSE_eval(y_true, y_pred):
    '''
    Beta-NSE metric
    
    -Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        
    -Returns:
        beta_nse: float, Beta-NSE decomposition
        
    References
    ----------
    .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error 
        and NSE performance criteria: Implications for improving hydrological modelling. Journal of hydrology, 377(1-2),
        80-91.
    '''
    
    return y_pred.mean() / y_true.mean()

def FHV_eval(y_true, y_pred, h=0.02):
    '''
    Calculate the peak flow bias of the flow duration curve
    
    -Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        h: float, fraction of upper flows to consider as peak flows of range (0,1), be default 0.02.
        
    -Returns:
        fhv: float, Peak flow bias
    
    References
    ----------
    .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process-based diagnostic approach to model 
        evaluation: Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, 
        doi:10.1029/2007WR006716. 
        
    '''
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    if y_true.shape[0] < 1:
        return np.nan
    
    if (h <= 0) or (h >= 1):
        raise ValueError("h has to be in range ]0,1[. Consider small values, e.g. 0.02 for 2% peak flows")
    
    # Get arrays sorted (descending) discharges
    y_true_sorted = np.sort(y_true)[::-1]
    y_pred_sorted = np.sort(y_pred)[::-1]

    # Subset data to only top h flow values
    y_true_subset = y_true_sorted[:int(h * len(y_true_sorted))]
    y_pred_subset = y_pred_sorted[:int(h * len(y_pred_sorted))]

    fhv = np.sum(y_pred_subset - y_true_subset) / np.sum(y_true_subset)
    
    return fhv * 100
    
def FMS_eval(y_true, y_pred, lower: float = 0.2, upper: float = 0.7):
    '''
    Calculate the slope of the middle section of the flow duration curve
    
    -Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        lower : float, optional, Lower bound of the middle section in range ]0,1[, by default 0.2
        upper : float, optional, Upper bound of the middle section in range ]0,1[, by default 0.7
        
    -Returns:
        fms: float, Slope of the middle section of the flow duration curve.
        
    References
    ----------
    .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process-based diagnostic approach to model
        evaluation: Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417,
        doi:10.1029/2007WR006716.
    '''
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Get arrays sorted (descending) discharges
    y_true_sorted = np.sort(y_true)[::-1]
    y_pred_sorted = np.sort(y_pred)[::-1]
    
    # For numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.   
    y_true_sorted[y_true_sorted <= 0] = 1e-6
    y_pred_sorted[y_pred_sorted <= 0] = 1e-6
    
    # Calculate FMS part by part
    qtm_lower = np.log(y_true_sorted[int(lower * len(y_true_sorted))])
    qtm_upper = np.log(y_true_sorted[int(upper * len(y_true_sorted))])
    qpm_lower = np.log(y_pred_sorted[int(lower * len(y_pred_sorted))])  
    qpm_upper = np.log(y_pred_sorted[int(upper * len(y_pred_sorted))])
    
    fms = ((qpm_lower - qpm_upper) - (qtm_lower - qtm_upper)) / (qtm_lower - qtm_upper + 1e-6)
    
    return fms * 100

def FLV_eval(y_true, y_pred, l=0.3):
    '''
    Calculate the low flow bias of the flow duration curv
    
    -Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        l: float, fraction of lower flows to consider as low flows of range (0,1), by default 0.3.
        
    -Returns:
        flv: float, Log-flow volume metric
        
    References
    ----------
    .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process-based diagnostic approach to model 
        evaluation: Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, 
        doi:10.1029/2007WR006716.     
    '''
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    if len(y_true) < 1:
        return np.nan
    
    if (l <= 0) or (l >= 1):
        raise ValueError("l has to be in range ]0,1[. Consider small values, e.g. 0.3 for 30% low flows")  
    
    # Get arrays sorted (descending) discharges
    y_true_sorted = np.sort(y_true)[::-1]
    y_pred_sorted = np.sort(y_pred)[::-1]
    
    # For numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.   
    y_true_sorted[y_true_sorted <= 0] = 1e-6
    y_pred_sorted[y_pred_sorted <= 0] = 1e-6
    
    y_true_sorted = np.log(y_true_sorted[-int(l * len(y_true_sorted)):])
    y_pred_sorted = np.log(y_pred_sorted[-int(l * len(y_pred_sorted)):])
    
    # Calculate FLV part by part
    qtl = np.sum(y_true_sorted - y_true_sorted.min())
    qpl = np.sum(y_pred_sorted - y_pred_sorted.min())
    
    flv = -1 * (qpl - qtl) / (qtl + 1e-6)
    
    return flv * 100

def KGE_eval(y_true, y_pred, weights = [1., 1., 1.]):
    '''
    Calculate the Kling-Gupta Efficieny
    
    - Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        weights: list, weights for the three components of the KGE, by default [1., 1., 1.]
        
    - Returns
        kge: float, Kling-Gupta Efficiency
        
    References
    ----------
    .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error 
        and NSE performance criteria: Implications for improving hydrological modelling. Journal of hydrology, 377(1-2),
        80-91.
    '''
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    if len(weights) != 3:
        raise ValueError("Weights of the KGE must be a list of three values")   
       
    if len(y_true) < 2:
        return np.nan
    
    r, _ = stats.pearsonr(y_true, y_pred)
    
    alpha = y_pred.std() / y_true.std()
    beta = y_pred.mean() / y_true.mean()
    
    value = ( (weights[0] * (r - 1))**2 + (weights[1] * (alpha - 1))**2 + (weights[2] * (beta - 1)) ** 2)
    
    return 1 - np.sqrt(value)

def betaKGE_eval(y_true, y_pred):
    '''
    Calculate the beta-KGE metric
    
    -Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        
    -Returns:
        beta_kge: float, Beta-KGE decomposition
        
    References
    ----------
    .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error 
        and NSE performance criteria: Implications for improving hydrological modelling. Journal of hydrology, 377(1-2),
        80-91.
        
    '''
        
    return y_pred.mean() / y_true.mean()

def MEAN_PEAK_TIMING_eval(y_true, y_pred, dates, window=3, resolution='1D'):
    '''
    Calculate the mean difference in peak flow timing
    
    -Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        dates: array_like, dates of the observations
        window: int, window size to consider around the peak, by default 3
        resolution: str, resolution of the dates, by default '1D'
    
    -Returns:
        mean_peak_timing_error: float, mean peak timing error
        
    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple 
        meteorological datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss., 
        https://doi.org/10.5194/hess-2020-221, in review, 2020.         
    '''
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    dates = dates.flatten()
    
    # heuristic to get indices of peaks and their corresponding height.
    peaks, _ = signal.find_peaks(y_true, distance=100, prominence=np.std(y_true))  
    
    # Evaluate timing
    timing_errors = []
    for idx in peaks:
    
        # Skip peaks at the start and end of the sequence and peaks around missing observations
        # (NaNs that were removed in obs & sim would result in windows that span too much time).
        if (idx - window < 0) or (idx + window >= len(y_true)) or np.arange(dates[idx - window], dates[idx + window + 1],        \
                            dtype='datetime64[' + resolution + ']').size != 2 * window + 1:
            continue
        
        # Check if the value at idx is a peak (both neighbors must be smaller)
        if (y_pred[idx] > y_pred[idx - 1]) and (y_pred[idx] > y_pred[idx + 1]):
            peak_pred = y_pred[idx]
        else:
            # Define peak around idx as the maximum value in the window
            peak_pred = np.max(y_pred[idx - window: idx + window + 1])
            
        # Get the corresponding peak in the simulation'
        peak_true = y_true[idx]
        
        # Calculate the time difference between the peaks
        # delta = np.abs(dates[idx] - dates[np.argmax(y_pred[idx - window: idx + window + 1])])
        date_idx_pred = np.where(np.abs(y_pred - peak_pred) < 1e-16)[0][0]
        date_idx_true = np.where(np.abs(y_true - peak_true) < 1e-16)[0]
        
        # Extract values from y_true closer to the indices in date_idx_pred
        date_idx_true = date_idx_true[np.argmin(np.abs(date_idx_true[:] - date_idx_pred))]
        
        delta = dates[date_idx_true] - dates[date_idx_pred]
        
        # Ensure delta is in the same units as the resolution
        delta = np.abs(delta.astype('timedelta64[' + resolution + ']'))
        
        timing_errors.append(delta)
        
    # timing_errors = timing_errors.astype('timedelta64[' + resolution + ']').view('int').astype(float)
    timing_errors = [t.astype('timedelta64[' + resolution + ']').view('int').astype(float) for t in timing_errors]
        
    return np.mean(timing_errors) if len(timing_errors) > 0 else np.nan

def MEAN_PEAK_MAPE_eval(y_true, y_pred):
    '''
    Calculate the mean absolute percentage error (MAPE) for peaks
    
    -Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        
    -Returns:
        peak_mape: float, mean absolute percentage peak error
        
    '''
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan  
    
    # Heuristic to get indices of peaks and their corresponding height.
    peaks, _ = signal.find_peaks(y_true, distance=100, prominence=np.std(y_true))
    
    # Check if there are any peaks
    if len(peaks) == 0:
        return np.nan
    
    # Subset data to only peaks
    y_true = y_true[peaks]
    y_pred = y_pred[peaks]
    
    # Calculate the mean absolute percentage peak error - MAPE
    peak_mape = np.sum(np.abs(y_true - y_pred) / y_true) / len(y_true) * 100
    
    return peak_mape

def PEARSON_R_eval(y_true, y_pred):
    '''
    Calculate pearson correlation coefficient (using scipy.stats.pearsonr)
    
    - Args:
        y_true: array_like, true values
        y_pred: array_like, predicted values
        
    - Returns:
        corr: float, Pearson correlation coefficient
    '''
    
    corr, _ = stats.pearsonr(y_true.flatten(), y_pred.flatten())
    
    return corr

def compute_all_metrics(y_true, y_pred, dates, metrics):
    
    metrics_dict = {}
    
    for metric in metrics:
        
        metric = metric.lower()
        
        try:
            if metric == 'peak-timing':
                metrics_dict[metric] = MEAN_PEAK_TIMING_eval(y_true, y_pred, dates)
            else:
                metrics_dict[metric] = metric_name_func_dict[metric](y_true, y_pred)
        except Exception as e:
            print(f"Error in computing metric {metric}: {e}")
            metrics_dict[metric] = np.nan
    
    return metrics_dict



metric_name_func_dict = {
    'nse': NSE_eval,
    'alpha-nse': alphaNSE_eval,
    'beta-nse': betaNSE_eval,
    'fhv': FHV_eval,
    'fms': FMS_eval,
    'flv': FLV_eval,
    'kge': KGE_eval,
    'beta-kge': betaKGE_eval,
    'peak-timing': MEAN_PEAK_TIMING_eval,
    'peak-mape':  MEAN_PEAK_MAPE_eval,
    'pearson-r': PEARSON_R_eval
}


if __name__ == "__main__":
    pass