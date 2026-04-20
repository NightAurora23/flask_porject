from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# Example dataset: [area, bedrooms]
X = np.array([
    [1000, 2],
    [1500, 3],
    [2000, 4],
    [2500, 4],
    [3000, 5]
])

# Prices
y = np.array([200000, 300000, 400000, 500000, 600000])

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open('house_price_model.pkl', 'wb'))
print("Model trained and saved!")
            