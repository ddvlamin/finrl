import pandas as pd
import unittest

from simulator import TradingSimulator

class TestTradingSimulator(unittest.TestCase):

    def setUp(self):
        # Initialize any resources or set up the environment before each test
        total_value = 1650.0
        self.positions = pd.DataFrame({
            "symbol": ["cash", "VW","VV"],
            "position": [100/total_value, 550/total_value, 1000/total_value],
            "price": [1, 110, 200],
            "shares": [100, 5, 5]}
        )
        self.positions.set_index("symbol", inplace=True)

    def tearDown(self):
        # Optional: Clean up resources or reset the environment after each test
        pass

    def test_reposition(self):
        simulator = TradingSimulator(self.positions)

        new_positions = self.positions.copy()
        new_positions.loc["cash", "position"] = 0.3
        new_positions.loc["VW", "position"] = 0.2
        new_positions.loc["VV", "position"] = 0.5

        simulator.reposition(new_positions)

        self.assertEqual(simulator.current_positions.loc["cash", "shares"], 520)
        self.assertEqual(simulator.current_positions.loc["VW", "shares"], 3)
        self.assertEqual(simulator.current_positions.loc["VV", "shares"], 4)

if __name__ == '__main__':
    unittest.main()
