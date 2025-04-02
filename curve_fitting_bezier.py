import numpy as np
import matplotlib.pyplot as plt
import re

def get_bezier_points():
    """
    Allows the user to mark 4 points interactively in a matplotlib plot.
    Returns the selected points as Bezier control points.
    """
    print("Please select 4 points in the plot (left-click). Close the plot window when done.")
    
    # Create an empty plot for user interaction
    fig, ax = plt.subplots()
    ax.set_title("Select 4 points for Bezier curve")
    ax.set_xlim(0, 20)  # Adjust limits as needed
    ax.set_ylim(0, 20)  # Adjust limits as needed
    ax.grid(True)
    
    # Use ginput to get 4 points from the user
    points = plt.ginput(4, timeout=0)  # Wait until 4 points are selected
    plt.close(fig)  # Close the plot after selection
    
    if len(points) != 4:
        raise ValueError("You must select exactly 4 points.")
    
    # Convert points to numpy array
    bezier_points = np.array(points)
    print("Selected Bezier control points:", bezier_points)
    
    return bezier_points

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

def construct_cubic_matrix_canon(t):
    cubic_terms = np.vstack([np.ones(len(t)), t, t**2, t**3]).T
    return cubic_terms

def extract_points_cubic_canon(Bx, By, n = 50):
    t = np.linspace(0, 1, n)  # Adjust range as needed
    
    # Calcular x e y usando o polinomio
    x = Bx[0] + Bx[1]*t + Bx[2]*t**2 + Bx[3]*t**3
    y = By[0] + By[1]*t + By[2]*t**2 + By[3]*t**3

    return x, y, t

def solve_linear_regression_fixed_points(matrix_t, points):
    # Pegar pontos referentes ao menor e maior t
    min_t_index = np.argmin(matrix_t[:,3])
    max_t_index = np.argmax(matrix_t[:,3])

    initial_point = points[min_t_index,:]
    final_point = points[max_t_index,:]

    factor_initial_point = matrix_t[:,0:1] * initial_point
    factor_final_point = matrix_t[:,-1:] * final_point

    matrix_t_no_endpoints = matrix_t[:, 1:-1]

    points_translated = points - factor_final_point - factor_initial_point
    coeficients_not_fixed = solve_linear_regression(matrix_t_no_endpoints, points_translated)

    final_coefficients = np.vstack([initial_point, coeficients_not_fixed, final_point])

    return final_coefficients 

def construct_linear_matrix_T(t):
    linear_terms = np.vstack([(1-t), (t)]).T
    return linear_terms

def construct_quadratic_matrix_T(t):
    quadratic_terms = np.vstack([(1-t)**2, (2*(1-t)*t), (t**2)]).T
    return quadratic_terms

def construct_cubical_matrix_T(t):
    cubic_terms = np.vstack([(1-t)**3, (3*((1-t)**2)*t), (3*(1-t)*(t**2)) , (t**3)]).T
    return cubic_terms

def solve_linear_regression(matrix_t, points):

    transform_y = matrix_t.T @ points
    coefficients = np.linalg.solve(matrix_t.T @ matrix_t, transform_y)
    return coefficients

def convert_from_bezier_coeff_to_canon(Bx, By):
    A = np.array([
        [ 1,  0,  0,  0],
        [-3,  3,  0,  0],
        [ 3, -6,  3,  0],
        [-1,  3, -3,  1]
    ])

    Bx = A @ Bx
    By = A @ By

    return Bx, By

def initialize_T(X, Y):
    T = [0]
    s = 0

    size_x = len(X)

    for i in range(1,size_x):
        s += dist(X[i], Y[i], X[i-1], Y[i-1])
        T.append(s)

    # T final
    T = np.array(T) / s
    return T

def update_T(T, Bx, By):

    # Primeira derivada
    d1 = np.array([
        [-3, 3, 0, 0],
        [ 0,-3, 3, 0],
        [ 0, 0,-3, 3]
    ])

    d1_Bx = d1 @ Bx.transpose()
    d1_By = d1 @ By.transpose()

    # Segunda derivada
    d2 = np.array([
        [  6,-12,  6,  0],
        [  0,  6,-12,  6],
    ])

    d2_Bx = d2 @ Bx.transpose()
    d2_By = d2 @ By.transpose()

    # T_cúbico
    T_cubico = construct_cubical_matrix_T(T)

    X1 = Bx @ T_cubico.transpose()
    Y1 = By @ T_cubico.transpose()

    # T_quadratico
    T_quadratico = construct_quadratic_matrix_T(T)

    X2 = d1_Bx @ T_quadratico.transpose()
    Y2 = d1_By @ T_quadratico.transpose()

    # T_linear
    T_linear = construct_linear_matrix_T(T)

    X3 = d2_Bx @ T_linear.transpose()
    Y3 = d2_By @ T_linear.transpose()

    # f(t)
    f = (X1 - X) * X2 + (Y1 - Y) * Y2

    # f'(t)
    d1_f = X2**2 + Y2**2 + (X1 - X) * X3 + (Y1 - Y) * Y3

    # t <- t - f(t)/f'(t)
    T -= f / d1_f

    T = (T - np.min(T)) / (np.max(T) - np.min(T))
    
    return T

read_points = False

if read_points:

    bezier_points = get_bezier_points()
    Bx = bezier_points[:, 0]
    By = bezier_points[:, 1]

    B_gt = np.vstack([Bx, By]).T
    X, Y, T_orig = extract_points_cubix(Bx, By, n=50) #extract_points(eps)

else:

    # Pegar os pontos
    with open('example.eps', 'r') as f:
        eps = f.read()

    X_pre, Y_pre = extract_points(eps)

    # Fixar 'n' para testar
    n = 30

    X = X_pre[2:n + 3]
    Y = Y_pre[2:n + 3]


T = initialize_T(X, Y)
points = np.vstack([X, Y]).T

for i in range(10):

    matrix_t = construct_cubical_matrix_T(T)
    extracted_coefficients = solve_linear_regression_fixed_points(matrix_t, points)

    T = update_T(T, extracted_coefficients[:,0], extracted_coefficients[:,1])
    
    points_extracted_from_cubix = extract_points_cubix(extracted_coefficients[:,0], extracted_coefficients[:,1], n=50)


points_extracted_from_cubix = extract_points_cubix(extracted_coefficients[:,0], extracted_coefficients[:,1], n=50)

plt.scatter(X, Y)
plt.plot(points_extracted_from_cubix[0], points_extracted_from_cubix[1], 'r')


if read_points:

    print(f'Error: {np.linalg.norm(B_gt - extracted_coefficients)}')
plt.show()
