import numpy as np
import matplotlib.pyplot as plt
import re
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Curve Fitting with Bezier")
    parser.add_argument("--read_points", type=bool, default=True, help="Whether to read points from file or generate new ones.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps for the fitting process.")
    parser.add_argument("--use_T_orig", type=bool, default=False, help="Whether to use the original T values.")
    parser.add_argument("--steps_newton", type=int, default=8, help="Number of Newton steps for updating T.")
    parser.add_argument("--path_to_bezier_points", type=str, default="bezier_points/b2.npy", help="Path to the Bezier points file.")
    return parser.parse_args()



def save_bezier_points(bezier_points, filename="bezier_points.npy"):
    """
    Saves the given Bezier points to a .npy file.

    Parameters:
        bezier_points (numpy.ndarray): The Bezier control points to save.
        filename (str): The name of the file to save the points to.
    """
    np.save(filename, bezier_points)
    print(f"Bezier points saved to {filename}")

def load_bezier_points(filename="bezier_points.npy"):
    """
    Loads Bezier points from a .npy file.

    Parameters:
        filename (str): The name of the file to load the points from.

    Returns:
        numpy.ndarray: The loaded Bezier control points.
    """
    bezier_points = np.load(filename)
    print(f"Bezier points loaded from {filename}")
    return bezier_points


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
    t = np.linspace(0, 1, n)  
    
    # Calcular x e y usando o polinomio
    x = Bx[0]*(1-t)**3 + Bx[1]*(3*((1-t)**2)*t)  + Bx[2] * (3*(1-t)*(t**2))  + Bx[3] * (t**3)
    y = By[0]*(1-t)**3 + By[1]*(3*((1-t)**2)*t)  + By[2] * (3*(1-t)*(t**2)) + By[3] * (t**3)

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

def compute_step_newton(T, Bx, By):
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


def update_T(T, Bx, By, steps_newton):

    for _ in range(steps_newton):
       
        T = compute_step_newton(T, Bx, By)
    
    return T


args = parse_arguments()

read_points = args.read_points
num_steps = args.num_steps
use_T_orig = args.use_T_orig
steps_newton = args.steps_newton
path_to_bezier_points = args.path_to_bezier_points


if read_points:

    if not os.path.exists(path_to_bezier_points):
        bezier_points = get_bezier_points()
        save_bezier_points(bezier_points, path_to_bezier_points)
    else:
        bezier_points = load_bezier_points(path_to_bezier_points)

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


if use_T_orig:
    T = T_orig
    num_steps = 1

else:   
    T = initialize_T(X, Y)


points = np.vstack([X, Y]).T




plt.ion()
fig1, ax1 = plt.subplots()  # Primeira janela

ax1.scatter(X, Y)


j = 1
Qt = 22
VErros = np.zeros(int((Qt - 1)/3))
XErrors = np.zeros(int((Qt - 1)/3))
print(VErros)
j2 = 0
while j < Qt:
    XErrors[j2] = j
    j2 += 1
    for i in range(j):
        matrix_t = construct_cubical_matrix_T(T)
        extracted_coefficients = solve_linear_regression_fixed_points(matrix_t, points)

        T = update_T(T, extracted_coefficients[:,0], extracted_coefficients[:,1], steps_newton)

    points_extracted_from_cubix = extract_points_cubix(extracted_coefficients[:,0], extracted_coefficients[:,1], n=50)

    ax1.scatter(X, Y) 
    ax1.plot(points_extracted_from_cubix[0], points_extracted_from_cubix[1], 'r')
    ax1.set_title(f'Iteração {j}')
    
    fig1.canvas.draw() 
    j += 3
    if read_points:
        print(f'Error: {np.linalg.norm(B_gt - extracted_coefficients)}')
    VErros[int((j - 1)/3) - 1] = np.linalg.norm(B_gt - extracted_coefficients)
    plt.pause(3)  # Tempo de espera entre os gráficos
    if j < Qt - 1:
        ax1.clear()
        
print(XErrors)
print(VErros)
plt.ioff()
fig2, ax2 = plt.subplots()  # Segunda janela
ax2.plot(XErrors, VErros,color='blue')
fig2.canvas.draw() 


plt.show()
