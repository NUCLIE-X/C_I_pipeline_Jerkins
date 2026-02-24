import pickle
import sys
import numpy as np
from sklearn.linear_model import LinearRegression

print("=== ML CI Pipeline ===")

# Create dataset
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

# Train model
model = LinearRegression()
model.fit(X, y)
print("Model trained")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved")

# Validate model
prediction = model.predict([[4]])[0]
print(f"Prediction: {prediction} (Expected: 8.0)")

if abs(prediction - 8.0) < 0.1:
    print("VALIDATION PASSED")
    sys.exit(0)
else:
    print("VALIDATION FAILED")
    sys.exit(1)
