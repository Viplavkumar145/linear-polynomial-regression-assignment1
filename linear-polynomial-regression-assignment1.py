#import required libraries first
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


np.random.seed(54) #Every time you run the code, you get the same output.
X = np.linspace(-3, 3, 100).reshape(-1, 1) # values of X 100 between -3, 3
y = X**3 - 3*X + np.random.normal(0, 3, size=X.shape) #cubic curve and noice

# lets split the data into train (70%) and test (30%)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=54
)

# Simple Linear Regression Model

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict the model
y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

# calculate the errors
mse_linear_train = mean_squared_error(y_train, y_train_pred)
mse_linear_test = mean_squared_error(y_test, y_test_pred)

print("Linear Regression Training MSE:", mse_linear_train)
print("Linear Regression Testing MSE:", mse_linear_test)


# Polynomial Regression

degrees = [1, 2, 3, 4, 8]  # degrees

train_erros = []  # Stores training error for each degree
test_erros = []   # Stores testing error for each degree

plt.figure(figsize=(10, 8))

for degree in degrees:
    # Create polynomial feature transformer
    poly = PolynomialFeatures(degree=degree)

    # Transform training and testing data
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Create linear regression model
    model = LinearRegression()

    # Train model on polynomial features
    model.fit(X_train_poly, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # calculate Mean Squared Error
    train_erros.append(mean_squared_error(y_train, y_train_pred))
    test_erros.append(mean_squared_error(y_test, y_test_pred))

    # curve for plotting
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    # Plot training data
    plt.scatter(X_train, y_train, color="blue", s=15)

    # Plot model prediction curve
    plt.plot(X_plot, y_plot, label=f"Degree {degree}")

# Display model prediction 

plt.title("Linear vs Polynomial Regression (Different Degrees)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Training errors vs Testing errors plot

plt.figure(figsize=(8, 5))

plt.plot(degrees, train_erros, marker='o', label="Training Error")
plt.plot(degrees, test_erros, marker='o', label="Testing Error")

plt.title("Training Error vs Testing Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()

# Conceptual Requirements

# Loss Function: Mean Squared Error (MSE)

# In linear regression, the loss function used is Mean Squared Error.
# The loss function tells us how wrong the model’s predictions are
# when compared to the real values.

# Squared error is used because it gives more importance to big mistakes.
# When the error is squared, large errors become much bigger than small ones.
# Squaring also removes negative values so that errors do not cancel each other.

# Minimizing the loss means making the model’s predictions as close as possible
# to the actual values. A smaller loss means the model is doing a better job.

# The model parameters affect the loss because they decide where the line or
# curve is placed. If the parameters are not correct, the predictions will be
# far from the real data and the loss will be high. When the parameters are
# adjusted properly, the loss becomes smaller and the model fits the data better.

# Analysis & Explanation

# WHY DOES TRAINING ERROR ALWAYS DECREASE WITH HIGHER POLYNOMIAL DEGREE?
# Training error decreases because higher-degree models are more flexible.
# When the degree increases, the model can bend more and fit the training data better.
# A simple model like degree 1 draws a straight line and misses the curve.
# A higher-degree model can follow the curve more closely.
# Because of this, the model makes fewer mistakes on training data as degree increases.

# WHY DOES TEST ERROR BEHAVE DIFFERENTLY?
# Test error does not always decrease.
# At first, test error decreases because the model is learning the real pattern.
# After a certain degree, test error increases because the model starts learning noise.
# This means the model works well on training data but poorly on new data.

# UNDERFITTING:
# Underfitting happens when the model is too simple.
# Low-degree models cannot capture the real relationship in the data.
# In underfitting, both training error and test error are high.
# Example: Degree 1 or 2 models.

# OVERFITTING:
# Overfitting happens when the model is too complex.
# High-degree models fit the noise in the training data.
# In overfitting, training error is very low but test error is high.
# Example: Degree 8 model.

# BIAS–VARIANCE TRADEOFF:
# Simple models have high bias and low variance.
# Complex models have low bias and high variance.
# The best model is in the middle, where bias and variance are balanced.
# This is seen where test error is minimum.

# FINAL OBSERVATION FROM THIS ASSIGNMENT:
# Degree 1: Underfitting – model is too simple.
# Degree 2: Still underfitting – curve is not captured fully.
# Degree 3: Best model – captures the real pattern well.
# Degree 4: Slight overfitting – learning extra details.
# Degree 8: Overfitting – learning noise instead of real pattern.
