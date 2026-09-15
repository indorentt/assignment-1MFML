import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    a = np.column_stack((A, b))

    print("\nAugmented Matrix:")
    print(a)

    # Forward Elimination with Partial Pivoting
    for i in range(n):
        # Partial pivoting
        max_row = i + np.argmax(abs(a[i:, i]))
        a[[i, max_row]] = a[[max_row, i]]

        if abs(a[i][i]) < 1e-10:
            raise ValueError("Zero pivot or singular matrix")

        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]
            a[j] = a[j] - factor * a[i]

    print("\nAfter Forward Elimination:")
    print(a)

    return a


def back_substitution(a):
    n = len(a)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if abs(a[i][i]) < 1e-10:
            raise ValueError("No unique solution")

        x[i] = (a[i][n] - np.dot(a[i][i+1:n], x[i+1:n])) / a[i][i]

    return x


def solve(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Invalid matrix dimensions")

    if len(b) != A.shape[0]:
        raise ValueError("Invalid RHS dimensions")

    print("\nCoefficient Matrix:")
    print(A)

    print("\nRHS Vector:")
    print(b)

    a = gaussian_elimination(A, b)

    x = back_substitution(a)

    print("\nSolution:")
    for i in range(len(x)):
        print("x{} = {:.2f}".format(i + 1, x[i]))

    # Verification
    print("\nVerification:")
    print("A × X =", np.dot(A, x))
    print("B     =", b)

    if np.allclose(np.dot(A, x), b):
        print("Solution Verified Successfully")


# -------- System 1 --------
A1 = [[2, 1, -1],
      [-3, -1, 2],
      [-2, 1, 2]]

b1 = [8, -11, -3]

print("========== SYSTEM 1 ==========")
solve(A1, b1)


# -------- System 2 --------
A2 = [[1, 1, 1],
      [2, 3, 1],
      [1, 2, 3]]

b2 = [6, 10, 13]

print("\n========== SYSTEM 2 ==========")
solve(A2, b2)