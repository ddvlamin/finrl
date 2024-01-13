from datetime import timedelta

import numpy as np
import cvxpy as cp
import pandas as pd


class TradingSimulator:
    def __init__(self, df, cash):
        """
        :param positions: dataframe indexed by symbol, also make sure to add a symbol for 'cash'
            and columns indicating symbol, current price and number of shared owned

            ```
            	symbol	price	shares
            	cash	0.01	10000
            	VW	    110	    5
            	VV	    200	    5
            ```

        :param cash: cash expressed in cents of the trading currency
        """
        self.df = df.copy()
        self.symbols = list(np.unique(self.df.index.get_level_values("symbol").values))

        self.current_tick = self.df.index.get_level_values("date").min().to_pydatetime()
        self.end_tick = self.df.index.get_level_values("date").max().to_pydatetime()

        self._init_positions(cash)
        self._next()

    def _init_positions(self, cash):
        open_prices = []
        shares = []
        for symbol in self.symbols:
            open_prices.append(self.df.loc[self.current_tick, symbol].loc["open"])
            shares.append(0)
        open_prices.append(0.01) # 1 share of cash is 0.01 worth in the trading currency
        shares.append(cash)

        self.current_positions = pd.DataFrame({"symbol": self.symbols + ["cash"], "price": open_prices, "shares": shares})
        self.current_positions.set_index("symbol", inplace=True)
        self.npositions = self.current_positions.shape[0]
        self._update()

    def _update(self):
        self.current_positions["total"] = self.current_positions["price"] * self.current_positions["shares"]
        self.total_value = self.current_positions["total"].sum()
        self.current_positions["position"] = self.current_positions["total"] / self.total_value

    def _next_tick(self, td=86400):
        one_day = timedelta(seconds=td)
        self.current_tick += one_day
        while self.current_tick.weekday() >= 5:
            self.current_tick += one_day

    def _next(self):
        self.current_stock = self.df.loc[self.current_tick].copy()
        self.current_positions.loc[self.current_stock["open"].index, "price"] = self.current_stock["open"]
        self._update()

    def reposition(self, positions):
        """
        :param positions: pd.Series with updated positions indexed by symbol

            ```
            	symbol	position
            	cash	0
            	VW	    0.7
            	VV	    0.3
            ```

        :return:
        """
        A = np.eye(self.npositions)
        b = np.ones((self.npositions,))
        for i, (symbol, row) in enumerate(self.current_positions.iterrows()):
            A[i, i] = row["price"] / self.total_value
            b[i] = positions.loc[symbol]
        #A[self.npositions-1, self.npositions-1] = self.current_positions.loc["cash", "price"] / self.total_value
        #b[self.npositions-1] = positions.loc["cash","position"]

        x = cp.Variable(self.npositions, integer=True)

        objective = cp.Minimize(cp.sum_squares(A @ x - b))
        prob = cp.Problem(objective, [cp.sum(A @ x) == 1, x >= 0])
        prob.solve()

        for i, (symbol, _) in enumerate(self.current_positions.iterrows()):
            self.current_positions.loc[symbol, "shares"] = int(np.round(x.value[i]))

        # if shares have been updated, we also need to update all derived variables
        self._update()