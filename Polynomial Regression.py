import numpy as np  # For numerical computations
import pandas as pd  # For handling datasets
import matplotlib.pyplot as plt  # For visualizations

# Load the dataset
data = pd.read_csv(r"D:\FSDS Material\Dataset\Non Linear emp_sal.csv")

# Extract independent variable (x) and dependent variable (y)
x = data.iloc[:, 1:2].values 
y = data.iloc[:, 2].values

# Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() # Initialize linear regression model
lin_reg.fit(x, y) # Fit the linear model to the data

# Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5) # Create polynomial features (degree 5)
x_poly = poly_reg.fit_transform(x) # Transform x to include polynomial terms 


poly_reg.fit(x_poly, y)  # Fit polynomial features to the target variable
lin_reg_2 = LinearRegression() # Initialize a second linear regression model
lin_reg_2.fit(x_poly, y) # Fit the transformed features

# Visualization: Linear Regression
plt.scatter(x, y, color = 'red') # Scatter plot of the original data
plt.plot(x, lin_reg.predict(x), color = 'blue') # Plot the linear regression line
plt.title(' Truth or Bluff (Linear Regression)') # Chart title
plt.xlabel('Position level') # x-axis label
plt.ylabel('Salary') # y-axis label
plt.show() # Display the plot

# Visualization: Polynomial Regression
plt.scatter(x, y, color='red')  # Scatter plot of the original data
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')  # Polynomial regression curve
plt.title('Truth or Bluff (Polynomial Regression)')  # Chart title
plt.xlabel('Position level')  # x-axis label
plt.ylabel('Salary')  # y-axis label
plt.show()  # Display the plot

# Predictions 

lin_model_pred = lin_reg.predict([[6.5]]) # Predict salary using linear model for position 6.5
lin_model_pred # Linear model prediction

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))  # Predict salary using polynomial model for position 6.5
poly_model_pred   # Polynomial model prediction

