import numpy as np
import matplotlib.pyplot as plt
import cvxopt

# Implementation of SVM using cvxopt
def svm(gamma: float, C: float, X: np.ndarray, y: np.ndarray):
    n_samples, _ = X.shape

    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j]) ** 2)

    # cvxopt maximizes 1/2 x^T P x + q^T x subject to Gx <= h and Ax = b
    q = cvxopt.matrix(-np.ones(n_samples))
    P = cvxopt.matrix(np.outer(y, y) * K)
    A = cvxopt.matrix(y, (1, n_samples))
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    with open("output.txt", "a") as f:
        f.write(f"γ = {gamma}, C = {C}, Optimal objective value: {-solution['primal objective']}\n")

    solution = np.array(solution["x"])
    support_vector_indices = np.where(solution > 1e-5)[0]
    a = np.array(solution).reshape(-1)
    b = 0
    for i in support_vector_indices:
        b += y[i]
        for j in support_vector_indices:
            b -= a[j] * y[j] * K[i, j]
    b /= len(support_vector_indices)

    return X[support_vector_indices], a[support_vector_indices] * y[support_vector_indices], b

def main():
    with open("data.txt", "r") as f:
        data = f.readlines()
    data = [list(map(float, x.strip().split())) for x in data]
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # SVM
    gammas = [10, 50, 100, 500]
    cs = [0.01, 0.1, 0.5, 1]

    for gamma in gammas:
        for C in cs:
            x_i, a, b = svm(gamma, C, X, y)
            print(f"Gamma: {gamma}, C: {C}, Support vectors: {x_i.shape[0]}")
            print(x_i.shape, a.shape)

            # Plot
            plt.figure()
            plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="red", label="1")
            plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color="blue", label="-1")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.title(f"Gamma: {gamma}, C: {C}")
            plt.legend()

            # Plot the decision boundary
            x1 = np.linspace(0, 1, 100)
            x2 = np.linspace(0, 1, 100)
            X1, X2 = np.meshgrid(x1, x2)
            Z = np.zeros_like(X1)
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    kernel = np.exp(-gamma * np.linalg.norm(x_i - np.array([X1[i, j], X2[i, j]]), axis=1) ** 2)
                    Z[i, j] = 1 if np.dot(a, kernel) + b >= 0 else -1

            # Add a heatmap
            plt.contourf(X1, X2, Z, alpha=0.3)
            plt.savefig(f"gamma_{gamma}_C_{C}.png")

if __name__ == "__main__":
    main()
