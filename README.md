# Logistic Regression from Scratch

This is a simple **Logistic Regression classifier implemented from scratch in Python** using NumPy.

## Features

- Binary classification
- Gradient descent optimization
- Custom log-likelihood calculation
- Predict probabilities or binary labels

## Usage

```python
from logistic_regression import Logistic_regression
import numpy as np

x_train = np.array([[1,2],[2,3],[3,4],[4,5]])
y_train = np.array([0,0,1,1])

model = Logistic_regression(num_iters=1000, alpha=0.1)
model.fit(x_train, y_train)

probs = model.predict(x_train)
labels = (probs > 0.5).astype(int)

print("Predicted probabilities:", probs)
print("Predicted labels:", labels)

