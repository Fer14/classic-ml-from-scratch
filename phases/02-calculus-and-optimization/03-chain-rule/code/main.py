import math

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


class Value:
    def __init__(self, data, parents=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self.parents = parents
        self.local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other):
        return Value(other) - self

    __radd__ = __add__
    __rmul__ = __mul__

    def exp(self):
        result = math.exp(self.data)
        return Value(result, (self,), (result,))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        return Value(s, (self,), (s * (1 - s),))

    def backward(self):
        ordered, visited = [], set()

        def visit(node):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node.parents:
                    visit(parent)
                ordered.append(node)

        visit(self)
        self.grad = 1.0
        for node in reversed(ordered):
            for parent, local in zip(node.parents, node.local_grads):
                parent.grad += local * node.grad


def logistic_loss_manual(w, b, x, y):
    z = w * x + b
    a = 1 / (1 + math.exp(-z))
    loss = -(y * math.log(a) + (1 - y) * math.log(1 - a))
    dloss_da = -(y / a) + (1 - y) / (1 - a)
    da_dz = a * (1 - a)
    dz_dw = x
    dz_db = 1.0
    dloss_dz = dloss_da * da_dz
    return loss, z, a, dloss_da, da_dz, dloss_dz, dloss_dz * dz_dw, dloss_dz * dz_db


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_gradients(weights, bias, features, labels):
    probabilities = sigmoid(features @ weights + bias)
    error = probabilities - labels
    return features.T @ error / len(labels), error.mean()


def logistic_loss(weights, bias, features, labels):
    probabilities = np.clip(sigmoid(features @ weights + bias), 1e-12, 1 - 1e-12)
    return -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))


def main():
    print("=== Step 1: Chain rule on a single composition ===")
    def outer(u):
        return math.sin(u)

    def inner(x):
        return x**2 + 1

    x0 = 0.7
    chain = math.cos(inner(x0)) * (2 * x0)
    numeric = central_difference(lambda x: outer(inner(x)), x0)
    print(f"  d/dx sin(x^2 + 1) at x = {x0}")
    print(f"  chain rule: cos(x^2 + 1) * 2x = {chain:.8f}")
    print(f"  numerical:                     {numeric:.8f}")

    print("\n=== Step 2: Backprop by hand through logistic loss ===")
    w, b, x, y = 0.5, -0.2, 2.0, 1.0
    loss, z, a, dloss_da, da_dz, dloss_dz, dloss_dw, dloss_db = logistic_loss_manual(w, b, x, y)
    print(f"  forward:  z = {z:.4f}, a = sigmoid(z) = {a:.4f}, loss = {loss:.4f}")
    print(f"  backward: dL/da = {dloss_da:.4f}, da/dz = {da_dz:.4f}")
    print(f"            dL/dz = dL/da * da/dz = {dloss_dz:.4f} (a - y = {a - y:.4f})")
    print(f"            dL/dw = dL/dz * x = {dloss_dw:.4f}, dL/db = {dloss_db:.4f}")
    numeric_w = central_difference(lambda value: logistic_loss_manual(value, b, x, y)[0], w)
    print(f"  numerical dL/dw = {numeric_w:.4f}")

    print("\n=== Step 3: Automatic differentiation with a tiny Value class ===")
    w_node, b_node = Value(0.5), Value(-0.2)
    a_node = (w_node * x + b_node).sigmoid()
    loss_node = -(y * a_node.log() + (1 - y) * (1 - a_node).log())
    loss_node.backward()
    print(f"  loss = {loss_node.data:.4f}, dL/dw = {w_node.grad:.4f}, dL/db = {b_node.grad:.4f}")

    print("\n=== Step 4: Gradient accumulation when a variable is reused ===")
    q = Value(3.0)
    r = q * q + q.exp() * q
    r.backward()
    print(f"  r = q^2 + q * e^q at q = 3, autodiff dr/dq = {q.grad:.6f}")
    print(f"  analytic 2q + e^q + q e^q = {2 * 3 + math.exp(3) + 3 * math.exp(3):.6f}")

    print("\n=== Step 5: Train logistic regression with chain-rule gradients ===")
    features, labels = make_classification(
        n_samples=500, n_features=4, n_informative=3, n_redundant=0, random_state=7
    )
    weights, bias = np.zeros(4), 0.0
    for epoch in range(3001):
        grad_w, grad_b = logistic_gradients(weights, bias, features, labels)
        weights -= 0.5 * grad_w
        bias -= 0.5 * grad_b
        if epoch % 1000 == 0:
            print(f"  epoch {epoch:4d} loss = {logistic_loss(weights, bias, features, labels):.5f}")
    print(f"  weights = {np.round(weights, 3)}, bias = {bias:.3f}")

    print("\n=== Use It: scikit-learn LogisticRegression ===")
    model = LogisticRegression(C=np.inf, max_iter=5000).fit(features, labels)
    print(f"  sklearn weights = {np.round(model.coef_[0], 3)}, bias = {model.intercept_[0]:.3f}")
    print(f"  sklearn loss = {logistic_loss(model.coef_[0], model.intercept_[0], features, labels):.5f}")
    agreement = np.mean((sigmoid(features @ weights + bias) > 0.5) == model.predict(features))
    print(f"  prediction agreement = {agreement:.3f}")


if __name__ == "__main__":
    main()
