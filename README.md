
# Quantum Machine Learning

This repository contains an example of using a parameterized quantum circuit as part of a machine learning workflow. The code demonstrates how to define a quantum circuit, train it using gradient descent, and make predictions on new data.

## Overview

The quantum circuit in this example is designed to be trained to minimize a cost function using the `GradientDescentOptimizer` from PennyLane. Once trained, the circuit is then used to make predictions on a testing dataset, and the performance is evaluated.

## Quantum Circuit

The quantum circuit is defined using PennyLane's quantum node (QNode) decorator. The circuit consists of rotation gates on two qubits, followed by a CNOT gate for entanglement, and additional rotation on the first qubit. The circuit returns the expectation values of the Pauli-Z observable on both qubits.

```python
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=0)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
```

## Cost Function

The cost function is defined as the sum of the expectation values returned by the circuit. The training process aims to find the set of parameters that minimizes this cost.

```python
def cost(params):
    return circuit(params)[0] + circuit(params)[1]
```

## Training Process

The training process involves initializing the parameters and using a classical optimizer to update these parameters to minimize the cost function.

```python
init_params = np.array([0.011, 0.012, 0.013], requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.4)
```

The training loop runs for a predetermined number of steps, updating the parameters at each step.

## Testing and Evaluation

After training, the circuit's performance is tested on a new dataset. The circuit makes predictions based on the test parameters, and the results are compared to the actual values using a mean squared error (MSE) metric.

```python
test_params = np.random.rand(5, 3)  # 5 sets of random test parameters
```

## Conclusion

This example  showcases how a quantum circuit can be used in a manner analogous to a classical machine learning model, using optimization techniques to train the model and make predictions.
