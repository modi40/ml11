import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("boston_houses.csv")

X = df[['RM']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train_poly, y_train)

y_pred = regressor.predict(X_test_poly)

print("Polynomial Regression Model (Degree 2) on Boston Housing:")
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
