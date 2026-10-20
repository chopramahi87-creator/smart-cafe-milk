"""cafe.py - The Milk Shop simulator (the BUSINESS in code)."""
import numpy as np
rng = np.random.default_rng()


class CafeSim:
    def __init__(self, price=5.0, order_cost=1.0, holding_cost=0.3,
                 spoilage_cost=1.5, stockout_penalty=6.0, spoil_frac=0.25,
                 mean_demand=7, busy_chance=0.20, max_stock=15, days=30):
        self.price = price; self.order_cost = order_cost
        self.holding_cost = holding_cost; self.spoilage_cost = spoilage_cost
        self.stockout_penalty = stockout_penalty; self.spoil_frac = spoil_frac
        self.mean_demand = mean_demand; self.busy_chance = busy_chance
        self.max_stock = max_stock; self.days = days

    def reset(self):
        self.stock = 5          # start each month with 5 units of milk on hand
        self.day = 0
        return self.stock       # STATE = stock on hand

    def _demand(self):
        d = rng.poisson(self.mean_demand)
        if rng.random() < self.busy_chance:      # busy day (weekend, hot day...)
            d += rng.poisson(5)
        return int(d)

    def step(self, order):
        order = int(np.clip(order, 0, self.max_stock))
        available = min(self.stock + order, self.max_stock)       # fridge limit
        demand = self._demand()
        sold = min(available, demand)                              # sell what you have
        stockout = max(demand - available, 0)                      # unhappy customers
        leftover = available - sold
        spoiled = int(round(self.spoil_frac * leftover))           # overnight spoilage
        carry = leftover - spoiled
        revenue = sold * self.price
        profit = revenue - (order * self.order_cost
                            + leftover * self.holding_cost
                            + spoiled * self.spoilage_cost
                            + stockout * self.stockout_penalty)
        self.stock = carry
        self.day += 1
        done = self.day >= self.days
        info = dict(order=order, demand=demand, sold=sold, spoiled=spoiled,
                    stockout=stockout, carry=carry, profit=round(profit, 2))
        return profit, self.stock, done, info


if __name__ == "__main__":
    env = CafeSim(); s = env.reset()
    for order in [8, 8, 2, 10]:
        r, s, done, info = env.step(order)
        print(info)
