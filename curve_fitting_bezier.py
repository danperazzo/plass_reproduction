import numpy as np
import matplotlib.pyplot as plt
import re

def extract_points(eps):
    X = []
    Y = []

    # Regular expression to match the coordinates before the 'p' command.
    points_pattern = r"(\d+(\.\d+)?)\s+(\d+(\.\d+)?) p"

    matches = re.findall(points_pattern, eps)
    for match in matches:
        X.append(float(match[0]))
        Y.append(float(match[2]))
    return np.array(X), np.array(Y)


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def extract_points_cubix(Bx, By, n = 50):
    t = np.linspace(0, 1, n)  # Adjust range as needed
    
    # Calcular x e y usando o polinomio
    x = Bx[0]*(1-t)**3 + Bx[1]*(3*((1-t)**2)*t)  + Bx[2] * (3*(1-t)*(t**2))  + Bx[3] * (t**3)
    y = By[0]*(1-t)**3 + By[1]*(3*((1-t)**2)*t)  + By[2] * (3*(1-t)*(t**2)) + By[3] * (t**3)

    return x, y, t

def construct_cubical_matrix_T(t):
    cubic_terms = np.vstack([(1-t)**3, (3*((1-t)**2)*t), (3*(1-t)*(t**2)) , (t**3)]).T
    return cubic_terms

def solve_linear_regression(matrix_t, points):

    transform_y = matrix_t.T @ points
    coefficients = np.linalg.solve(matrix_t.T @ matrix_t, transform_y)
    return coefficients


# Pegar os pontos
with open('example.eps', 'r') as f:
    eps = f.read()


Bx = np.array([  0,  0,  10,10])
By = np.array([  0,   10, 10,0 ])
B_gt = np.vstack([Bx, By]).T

n = 10
X, Y, T_orig = extract_points_cubix(Bx, By, n=n) #extract_points(eps)

T = [0]
s = 0

for i in range(1,n):
    s += dist(X[i], Y[i], X[i-1], Y[i-1])
    T.append(s)

# T final
T = np.array(T) / s
print(T)
points = np.vstack([X, Y]).T
#T = T_orig

matrix_t = construct_cubical_matrix_T(T)

print(matrix_t)

#polynomial_matrix = construct_cubical_matrix_T(T)


extracted_coefficients = solve_linear_regression(matrix_t, points)
print(extracted_coefficients)
print(B_gt)

points_extracted_from_cubix = extract_points_cubix(extracted_coefficients[:,0], extracted_coefficients[:,1], n=50)

print(f'error: {np.linalg.norm(B_gt - extracted_coefficients)}')

plt.scatter(X, Y)
plt.plot(points_extracted_from_cubix[0], points_extracted_from_cubix[1], 'r')

plt.show()

