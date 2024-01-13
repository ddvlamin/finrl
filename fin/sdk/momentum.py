import pandas as pd
import numpy as np
from scipy.stats import kendalltau

def ribbon(df, column, range):
    df_dict = {}
    for window in range:
        df_dict[f"{column}_ma_{window}"] = df[column].rolling(window=window).mean()
    return pd.DataFrame(df_dict)

# Function to calculate Kendall Tau distance for a row
def kendalltau_distance(row, n_ribbons):
    x = np.arange(n_ribbons, 0, -1) # monotonic decreasing row
    tau = kendalltau(x, row)
    return 1-(tau.statistic+1)/2

def ribbon_kendalltau_distance(df):
    """
    The distance to the perfect ranking which is a monotonic decreasing row.
    So if the first column is a short-period moving average and the last column a long-period moving average
    then a distance of zero indicates a buy signal, if the distance is one it indicates a sell signal

    :param df:
    :return:
    """
    return df.apply(lambda row: kendalltau_distance(row, df.shape[1]), axis=1)

def threshold_signal(tseries, threshold, is_lower, value):
    tseries = tseries.copy()
    if is_lower:
        tseries[tseries < threshold] = value
        tseries[tseries >= threshold] = np.nan
    else:
        tseries[tseries > threshold] = value
        tseries[tseries <= threshold] = np.nan
    return tseries

def kendalltau_buy_sell_triggers(tseries, buy_threshold, sell_threshold):
    buy = threshold_signal(tseries, buy_threshold, True, 0)
    sell = threshold_signal(tseries, sell_threshold, False, 1)

    buy_sell = buy.combine_first(sell)

    buy_sell = buy_sell.fillna(method='ffill')
    buy_sell.loc[buy_sell == buy_sell.shift()] = np.nan

    buy = buy_sell.loc[buy_sell == 0]
    sell = buy_sell.loc[buy_sell == 1]

    return buy_sell, buy, sell


def ribbon_correlation(column, ribbons, window_size):
    """
    high correlation with low variance means either
    :param column:
    :param ribbons:
    :param window_size:
    :return:
    """
    df_dict = {}
    for ribbon_name in ribbons.columns:
        if ribbon_name != column:
            ribbon = ribbons[ribbon_name]
            df_dict[f"{ribbon_name}_corr"] = ribbons[column].rolling(window=window_size).corr(ribbon)
    df = pd.DataFrame(df_dict)
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    df[f"{column}_ma_corr_mean"] = df_mean
    df[f"{column}_ma_corr_std"] = df_std
    return df