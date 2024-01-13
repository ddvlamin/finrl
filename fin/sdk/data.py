import requests
import os
import pandas as pd
import certifi
from urllib.request import urlopen
import json
import numpy as np

BASE_URL = "https://financialmodelingprep.com/api/v3"
API_KEY = os.environ["FIN_MODEL_PREP_KEY"]

def from_date(df, start_date):
    return df.loc[(slice(start_date, None), slice(None)), :]

def select_symbol(df, symbol):
    return df.loc[(slice(None), symbol), :].reset_index(level='symbol', drop=True)

def _convert_unix_timestamp(df, column):
    return pd.to_datetime(df[column], unit='s')

def _read_response(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return pd.DataFrame(json.loads(data))

def get_historical_stock_df(period, symbols, from_date, to_date):
    dfs = []
    for symbol in symbols:
        url = f"{BASE_URL}/historical-chart/{period}/{symbol}?from={from_date}&to={to_date}&apikey={API_KEY}"
        df = _read_response(url)

        df['date'] = pd.to_datetime(df['date'])
        df["symbol"] = symbol

        df.set_index(['date','symbol'], inplace=True)

        dfs.append(df)

    df = pd.concat(dfs)
    df = _forward_fill_historical_stock(df)

    return df

def _forward_fill_historical_stock(df, date_index_name='date', symbol_index_name='symbol'):
    dts = df.index.get_level_values(date_index_name)
    date_range = pd.date_range(start=dts.min(), end=dts.max(), freq='D')

    # Reindex the DataFrame with the complete date range
    new_index = pd.MultiIndex.from_product([date_range,
                                            df.index.get_level_values(symbol_index_name).unique()],
                                           names=[date_index_name, symbol_index_name])
    df = df.reindex(new_index).groupby(symbol_index_name).ffill()

    return df

def get_stock_df():
    url = f"{BASE_URL}/available-traded/list?apikey={API_KEY}"
    df = _read_response(url)
    df = df[df["type"] == "stock"]
    return df

def get_exchanges():
    stocks_df = get_stock_df()
    return set(stocks_df["exchange"].unique())

def get_exchange_symbols(exchange):
    stock_df = get_stock_df()
    stock_df = stock_df[stock_df["exchange"] == exchange]
    return set(stock_df["symbol"].unique())

def get_cash_flow_statements(symbol, period="annual"):
    url = f"{BASE_URL}/cash-flow-statement/{symbol}?period={period}&apikey={API_KEY}"
    df = _read_response(url)
    return df

def predict_average_free_cash_flow_growth(df):
    pass

def estimate_dcf():
    """
    https://site.financialmodelingprep.com/developer/docs/dcf-formula

    https://www.ilpabogados.com/en/valuation-of-discounted-cash-flows/

    :return:
    """
    pass