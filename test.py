from lsr import LSR
import pandas as pd
import matplotlib.pyplot as plt

X_treino = [1, 2, 3, 4, 5]
y_treino = [2, 4, 5, 4, 5]

formatted_data = list(zip(X_treino, y_treino))

model = LSR(data_points=formatted_data)
model.fit()

model.plot()
