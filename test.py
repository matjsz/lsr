from lsr import LSR
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/house_prices_practice.csv").sample(n=50)

X = 'LotArea'
Y = 'SalePrice'

formatted_data = list(zip(df[X].to_list(), df[Y].to_list()))

df.plot.scatter(X, Y)
plt.show()

model = LSR(data_points=formatted_data)
model.fit()

model.plot()