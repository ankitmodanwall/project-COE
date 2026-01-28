import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample house dataset
data = {
    'Area': [800, 1000, 1200, 1500, 1800],
    'Bedrooms': [1, 2, 2, 3, 3],
    'Age': [10, 8, 6, 5, 2],
    'Price': [3000000, 4000000, 5000000, 6500000, 8000000]
}

df = pd.DataFrame(data)


X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)


def predict_price(area, bedrooms, age):
    price = model.predict([[area, bedrooms, age]])
    return price[0]


area = float(input("Enter house area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
age = int(input("Enter age of house (years): "))

predicted_price = predict_price(area, bedrooms, age)

print(f"\n🏠 Predicted House Price: ₹{predicted_price:,.2f}")
