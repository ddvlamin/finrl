import requests
import os
import pandas as pd
import certifi
from urllib.request import urlopen
import json
import numpy as np

BASE_URL = "https://financialmodelingprep.com/api/v3"
API_KEY = os.environ["FIN_MODEL_PREP_KEY"]

def _convert_unix_timestamp(df, column):
    return pd.to_datetime(df[column], unit='s')

def _read_response(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return pd.DataFrame(json.loads(data))

def get_historical_stock_df(period, symbol, from_date, to_date):
    url = f"{BASE_URL}/historical-chart/{period}/{symbol}?from={from_date}&to={to_date}&apikey={API_KEY}"
    df = _read_response(url)
    df['date'] = pd.to_datetime(df['date'])
    df = _forward_fill_historical_stock(df)
    #df.set_index('date', inplace=True)

    return df

def _forward_fill_historical_stock(df):
    date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

    # Reindex the DataFrame with the complete date range
    df = df.set_index('date').reindex(date_range)
    #df = df.reset_index(drop=True)
    #df = df.rename(columns={'index': 'date'})

    # Forward fill the missing values with the last known value
    df['open'] = df['open'].ffill()
    df['high'] = df['high'].ffill()
    df['low'] = df['low'].ffill()
    df['close'] = df['close'].ffill()
    df['volume'] = df['volume'].ffill()

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