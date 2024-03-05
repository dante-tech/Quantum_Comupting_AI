import pennylane as qml
from pennylane import numpy as np

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=0)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

def cost(params):
    return circuit(params)[0] + circuit(params)[1]

# Define a true model with fixed parameters for comparison
true_params = np.array([0.05, -0.02, 0.10])

# Function to calculate mean squared error
def mse(predictions, actuals):
    return np.mean((np.array(predictions) - np.array(actuals))**2)

test_params = np.random.rand(5, 3)

all_predictions = []
all_actuals = []
for i, test_param_set in enumerate(test_params):
    prediction = circuit(test_param_set)
    actual = circuit(true_params)
    test_mse = mse(prediction, actual)
    all_predictions.append(prediction)
    all_actuals.append(actual)
    print(f"Test MSE for dataset {i+1}: {test_mse:.4f}")
    print(f"Prediction for dataset {i+1}: {prediction}")
    print(f"Actual for dataset {i+1}: {actual}")

# Calculate overall MSE for all test datasets
overall_mse = mse(all_predictions, all_actuals)
print(f"Overall MSE for all test datasets: {overall_mse:.4f}")
