import numpy as np
import pandas as pd

from scipy import stats

def compute_slopes(ts, window=30, pvalue_threshold=0.0000001):
    slopes = np.zeros((ts.shape[0],))
    p_values = np.zeros((ts.shape[0],))
    for i in range(ts.shape[0] - window):
        xi_window = np.arange(window)
        y_window = ts.values[i:i + window]
        slope, intercept, r_value, p_value, std_err = stats.linregress(xi_window, y_window)
        slopes[i + window - 1] = slope
        p_values[i + window - 1] = p_value

    significant_slopes = np.zeros((ts.shape[0],))
    significant_slopes[p_values < pvalue_threshold] = slopes[p_values < pvalue_threshold]

    return significant_slopes

def compute_slopes_per_symbol(df, window=30, pvalue_threshold=0.0000001):
    df_transposed = df.unstack()
    slopes = dict()
    for symbol in df_transposed.columns:
        symbol_slopes = compute_slopes(df_transposed[symbol], window, pvalue_threshold)
        slopes[symbol] = symbol_slopes

    dts = df.index.get_level_values("date")
    date_range = pd.date_range(start=dts.min(), end=dts.max(), freq='D')
    slopes["date"] = date_range

    slopes_df = pd.DataFrame(slopes)
    slopes_df.set_index(['date'], inplace=True)

    return slopes_df.stack()