# import required libraries first
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


np.random.seed(54)  # Every time you run the code, you get the same output.
X = np.linspace(-3, 3, 100).reshape(-1, 1)  # 100 values of X between -3 and 3
y = X**3 - 3*X + np.random.normal(0, 3, size=X.shape)  # cubic curve with noise

# Split data into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=54
)

# Simple Linear Regression Model

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_train_pred_linear = linear_model.predict(X_train)
y_test_pred_linear  = linear_model.predict(X_test)

mse_linear_train = mean_squared_error(y_train, y_train_pred_linear)
mse_linear_test  = mean_squared_error(y_test,  y_test_pred_linear)

print("Linear Regression Training MSE:", mse_linear_train)
print("Linear Regression Testing MSE: ", mse_linear_test)

# Polynomial Regression (multiple degrees)

degrees = [1, 2, 3, 4, 8]  # at least 4 different degrees as required

train_errors = []
test_errors  = []

plt.figure(figsize=(12, 7))

# Plot both training AND test data so under/overfitting is clearly visible
plt.scatter(X_train, y_train, color="blue",  s=15, label="Training Data", zorder=3)
plt.scatter(X_test,  y_test,  color="green", s=15, label="Testing Data",  zorder=3, alpha=0.6)

X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)  # smooth curve for plotting

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly  = poly.transform(X_test)
    X_plot_poly  = poly.transform(X_plot)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred  = model.predict(X_test_poly)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test,  y_test_pred))

    y_plot = model.predict(X_plot_poly)
    plt.plot(X_plot, y_plot, label=f"Degree {degree}")

plt.title("Linear vs Polynomial Regression (Different Degrees)")
plt.xlabel("X")
plt.ylabel("y")
plt.ylim(-25, 25)   # clip extreme overfitting curves so the data stays visible
plt.legend()
plt.tight_layout()
plt.show()

# Training Error vs Testing Error Plot

plt.figure(figsize=(8, 5))

plt.plot(degrees, train_errors, marker='o', label="Training Error")
plt.plot(degrees, test_errors,  marker='o', label="Testing Error")

# Use the actual degree values as x-tick labels (not evenly spaced integers)
plt.xticks(degrees)

plt.title("Training Error vs Testing Error by Polynomial Degree")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend()
plt.tight_layout()
plt.show()

# Print a summary table for easy reading
print("\n{:<10} {:>18} {:>18}".format("Degree", "Train MSE", "Test MSE"))
print("-" * 48)
for d, tr, te in zip(degrees, train_errors, test_errors):
    print("{:<10} {:>18.4f} {:>18.4f}".format(d, tr, te))


# Conceptual Requirements

# Loss Function: Mean Squared Error (MSE)
#
# In linear regression the loss function is Mean Squared Error (MSE).
# The formula is: MSE = (1/n) * Σ (y_i - ŷ_i)²

# Why squared errors?
    
#   Squaring the difference between each actual value (y_i) and predicted
#   value (ŷ_i) serves two purposes:
#   1. It makes all errors positive, so positive and negative errors do not
#      cancel each other out.
#   2. It penalises large errors much more heavily than small ones, forcing
#      the model to avoid big mistakes rather than settling for many small ones.
#
# WHAT DOES MINIMISING THE LOSS MEAN?
#   Minimising MSE means finding model parameters (slope and intercept) that
#   bring the predicted line as close as possible to every data point on
#   average. A lower MSE means the model's guesses are closer to reality.
#
# HOW DO MODEL PARAMETERS AFFECT THE LOSS?
#   The parameters (weights/coefficients) decide the position and shape of the
#   regression line or curve. If the parameters are wrong, predictions will be
#   far from the actual values and MSE will be high. When we adjust parameters
#   (e.g. via the Normal Equation or gradient descent) to minimise MSE, the
#   line shifts until it best fits the data.

# ANALYSIS & EXPLANATION

# WHY DOES TRAINING ERROR ALWAYS DECREASE WITH HIGHER POLYNOMIAL DEGREE?
#   Each extra degree gives the model more freedom to bend and twist.
#   A degree-1 model is a straight line — it cannot follow a cubic curve.
#   A degree-8 model has enough flexibility to pass almost through every
#   training point. Because we measure training error ON THE SAME DATA the
#   model was trained on, a more flexible model will always fit that data
#   better, driving training MSE down (and toward 0 at very high degrees).

# WHY DOES TEST ERROR BEHAVE DIFFERENTLY?
#   The test set contains data the model has never seen. A model that
#   memorises noise in the training data will make poor predictions on new
#   points. So while training error keeps falling, test error first decreases
#   (the model is learning the true pattern) and then rises again (the model
#   has started fitting noise instead of the pattern).

# AT WHAT POINT DOES THE MODEL START OVERFITTING, AND HOW CAN YOU TELL?
#   From the Training vs Testing Error plot, the test error reaches its lowest
#   point around degree 3 and increases noticeably from degree 4 onward.
#   The clearest sign of overfitting is the gap: training error is very low
#   while test error is significantly higher. At degree 8 the prediction curve
#   also shows wild swings near the edges of the data range — a classic visual
#   sign of overfitting.

# WHICH POLYNOMIAL DEGREE WOULD YOU CHOOSE, AND WHY?
#   Degree 3 is the best choice.
#   - It matches the true underlying function (x³ - 3x), so it is not
#     coincidence that degree 3 gives the lowest or near-lowest test error.
#   - Training error and test error are close to each other, meaning the model
#     generalises well to unseen data.
#   - The fitted curve on the scatter plot follows the overall trend without
#     wild swings, confirming a good bias–variance balance.

# UNDERFITTING (Degrees 1 and 2):
    
#   The model is too simple to capture the S-shaped cubic pattern.
#   Both training and test errors are high.
#   The fitted line misses the curves in the data.

# OVERFITTING (Degrees 4 and especially 8):
#   The model is too complex and memorises noise.
#   Training error is very low but test error rises.
#   The prediction curve shows large oscillations, especially at the edges.

# BIAS–VARIANCE TRADEOFF:
#   Low-degree models → high bias (wrong assumptions), low variance (stable).
#   High-degree models → low bias (flexible), high variance (sensitive to noise).
#   Degree 3 sits at the sweet spot where the test error is minimised,
#   balancing bias and variance for the best generalisation.
