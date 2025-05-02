import numpy as np
import matplotlib
#matplotlib.use('TkAgg')  # <-- adiciona isso no início
import matplotlib.pyplot as plt
import re
import os
import argparse
import matplotlib.animation as animation

import cvxpy as cp
import scipy.optimize as opt

def fit_bezier_tangent(t_data, X_data, Y_data, x_0, x_m, y_0, y_m, vx, vy, position_gradient):
    """
    Fits a cubic Bézier curve segment (simplified version).
    Assumes inputs are valid NumPy arrays and position_gradient is 'left' or 'right'.
    Finds Bx, By, and alpha using Linear Least Squares.

    Args:
        t_data (np.ndarray): Array of parameter values 't'.
        X_data (np.ndarray): Array of observed X coordinates.
        Y_data (np.ndarray): Array of observed Y coordinates.
        x_0, x_m, y_0, y_m (float): Start/end point coordinates.
        vx, vy (float): Tangent vector components.
        position_gradient (str): 'left' or 'right'.

    Returns:
        tuple: (Bx_fit, By_fit, alpha_fit)
    """
    N = len(t_data)
    A = np.zeros((2 * N, 3))
    b = np.zeros(2 * N)

    # Pre-calculate powers of t and (1-t)
    t = t_data # Rename for brevity
    t_1 = 1 - t
    t_sq = t**2
    t_1_sq = t_1**2
    t_cub = t**3
    t_1_cub = t_1**3

    # Calculate common coefficient terms based on Bernstein polynomials
    coeff_3t_1_t_sq = 3 * t * t_1_sq
    coeff_3t_sq_1_t = 3 * t_sq * t_1

    if position_gradient == 'left':
        # Target: p = [Bx_1, By_1, alpha]
        KnownX = x_0 * t_1_cub + x_m * coeff_3t_sq_1_t + x_m * t_cub
        KnownY = y_0 * t_1_cub + y_m * coeff_3t_sq_1_t + y_m * t_cub

        Coeff_B1_common = coeff_3t_1_t_sq # Coefficient for Bx_1 and By_1
        Coeff_AlphaX = (-vx / 3.0) * coeff_3t_sq_1_t
        Coeff_AlphaY = (-vy / 3.0) * coeff_3t_sq_1_t

        A[0::2, 0] = Coeff_B1_common
        A[0::2, 2] = Coeff_AlphaX
        A[1::2, 1] = Coeff_B1_common
        A[1::2, 2] = Coeff_AlphaY

    elif position_gradient == 'right': # Assuming only 'left' or 'right'
        # Target: p = [Bx_2, By_2, alpha]
        KnownX = x_0 * t_1_cub + x_0 * coeff_3t_1_t_sq + x_m * t_cub
        KnownY = y_0 * t_1_cub + y_0 * coeff_3t_1_t_sq + y_m * t_cub

        Coeff_B2_common = coeff_3t_sq_1_t # Coefficient for Bx_2 and By_2
        Coeff_AlphaX = (vx / 3.0) * coeff_3t_1_t_sq
        Coeff_AlphaY = (vy / 3.0) * coeff_3t_1_t_sq

        A[0::2, 0] = Coeff_B2_common
        A[0::2, 2] = Coeff_AlphaX
        A[1::2, 1] = Coeff_B2_common
        A[1::2, 2] = Coeff_AlphaY

    # Construct vector b (target residuals)
    b[0::2] = X_data - KnownX
    b[1::2] = Y_data - KnownY

    # Solve the linear system A * p = b
    # np.linalg.lstsq returns: params, residuals, rank, singular_values
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return params[0], params[1], params[2] # Bx_fit, By_fit, alpha_fit


def fit_bezier_tangents_simultaneous(t_data, X_data, Y_data, x_0, x_m, y_0, y_m, vx, vy):
    """
    Fits a cubic Bézier curve segment with fixed start/end tangents directions.
    Assumes inputs are valid NumPy arrays.
    Finds alpha_l and alpha_r simultaneously using Linear Least Squares.

    Args:
        t_data (np.ndarray): Array of parameter values 't'.
        X_data (np.ndarray): Array of observed X coordinates.
        Y_data (np.ndarray): Array of observed Y coordinates.
        x_0, x_m, y_0, y_m (float): Start/end point coordinates.
        vx (np.ndarray or list/tuple): Tangent X components [vx_l, vx_r].
        vy (np.ndarray or list/tuple): Tangent Y components [vy_l, vy_r].
                                     (Note: Corrected usage from user's error fn)

    Returns:
        tuple: (alpha_l_fit, alpha_r_fit)
    """
    N = len(t_data)
    A = np.zeros((2 * N, 2))
    b = np.zeros(2 * N)

    # Ensure vx, vy are accessible by index
    vx_l, vx_r = vx[0], vx[1]
    vy_l, vy_r = vy[0], vy[1] # Corrected based on Y_LR definition

    # Pre-calculate powers of t and (1-t)
    t = t_data # Rename for brevity
    t_1 = 1 - t
    t_sq = t**2
    t_1_sq = t_1**2
    t_cub = t**3
    t_1_cub = t_1**3

    # Calculate common coefficient terms based on Bernstein polynomials
    coeff_3t_1_t_sq = 3 * t * t_1_sq  # Coeff for B1
    coeff_3t_sq_1_t = 3 * t_sq * t_1  # Coeff for B2

    # Calculate known parts of the curve (without the alpha terms)
    KnownX = x_0 * t_1_cub + x_0 * coeff_3t_1_t_sq + x_m * coeff_3t_sq_1_t + x_m * t_cub
    KnownY = y_0 * t_1_cub + y_0 * coeff_3t_1_t_sq + y_m * coeff_3t_sq_1_t + y_m * t_cub

    # Calculate coefficients for alpha_l and alpha_r
    Coeff_AlphaL_X = (-vx_l / 3.0) * coeff_3t_sq_1_t
    Coeff_AlphaR_X = ( vx_r / 3.0) * coeff_3t_1_t_sq
    Coeff_AlphaL_Y = (-vy_l / 3.0) * coeff_3t_sq_1_t
    Coeff_AlphaR_Y = ( vy_r / 3.0) * coeff_3t_1_t_sq

    # Construct matrix A (shape 2N x 2)
    A[0::2, 0] = Coeff_AlphaL_X  # Column for alpha_l, X equations
    A[0::2, 1] = Coeff_AlphaR_X  # Column for alpha_r, X equations
    A[1::2, 0] = Coeff_AlphaL_Y  # Column for alpha_l, Y equations
    A[1::2, 1] = Coeff_AlphaR_Y  # Column for alpha_r, Y equations

    # Construct vector b (target residuals)
    b[0::2] = X_data - KnownX
    b[1::2] = Y_data - KnownY

    # Solve the linear system A * p = b, where p = [alpha_l, alpha_r]
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return params[0], params[1] # alpha_l_fit, alpha_r_fit


def subdivide_cubic_bezier(Bx, By):
    """
    Subdivide a cubic Bézier curve at t=0.5 into two curves.

    Parameters:
        Bx (np.ndarray): A 1D array of x-coordinates (shape (4,))
        By (np.ndarray): A 1D array of y-coordinates (shape (4,))

    Returns:
        B1x, B1y, B2x, B2y: The control points for the left and right sub-curves.
            - Left curve control points: (Bx[0], By[0]), Q0, R0, S0
            - Right curve control points: S0, R1, Q2, (Bx[3], By[3])
    """
    # Level 1: Compute the midpoints between consecutive original control points.
    Q0x = (Bx[0] + Bx[1]) / 2.0
    Q1x = (Bx[1] + Bx[2]) / 2.0
    Q2x = (Bx[2] + Bx[3]) / 2.0

    Q0y = (By[0] + By[1]) / 2.0
    Q1y = (By[1] + By[2]) / 2.0
    Q2y = (By[2] + By[3]) / 2.0

    # Level 2: Compute the midpoints between the Q points.
    R0x = (Q0x + Q1x) / 2.0
    R1x = (Q1x + Q2x) / 2.0

    R0y = (Q0y + Q1y) / 2.0
    R1y = (Q1y + Q2y) / 2.0

    # Level 3: Compute the midpoint between the R points (the point on the curve at t=0.5).
    S0x = (R0x + R1x) / 2.0
    S0y = (R0y + R1y) / 2.0

    # Create the left sub-curve:
    # Control points: P0, Q0, R0, S0.
    B1x = np.array([Bx[0], Q0x, R0x, S0x])
    B1y = np.array([By[0], Q0y, R0y, S0y])

    # Create the right sub-curve:
    # Control points: S0, R1, Q2, P3.
    B2x = np.array([S0x, R1x, Q2x, Bx[3]])
    B2y = np.array([S0y, R1y, Q2y, By[3]])

    # Compute the derivative at t=0.5:
    # For a cubic Bézier curve, B'(0.5) = 3 * (R1 - R0)
    dSx = (R1x - R0x)
    dSy =  (R1y - R0y)

    return B1x, B1y, B2x, B2y, dSx, dSy


def _de_casteljau(Bx, By, t):
    """
    Run one full de Casteljau on the 4 pts P0,P1,P2,P3 at param t.
    Returns:
        Q0, Q1, Q2  (each as (x,y))
        R0, R1      (each as (x,y))
        S           (x,y)  — the point on the curve at t
    """
    # Level 1
    Q0 = ((1-t)*Bx[0] + t*Bx[1], (1-t)*By[0] + t*By[1])
    Q1 = ((1-t)*Bx[1] + t*Bx[2], (1-t)*By[1] + t*By[2])
    Q2 = ((1-t)*Bx[2] + t*Bx[3], (1-t)*By[2] + t*By[3])

    # Level 2
    R0 = ((1-t)*Q0[0] + t*Q1[0], (1-t)*Q0[1] + t*Q1[1])
    R1 = ((1-t)*Q1[0] + t*Q2[0], (1-t)*Q1[1] + t*Q2[1])

    # Level 3: point on curve
    S  = ((1-t)*R0[0] + t*R1[0], (1-t)*R0[1] + t*R1[1])

    return Q0, Q1, Q2, R0, R1, S


def subdivide_cubic_bezier_at_two_ts(Bx, By, t1, t2):
    """
    Subdivide the cubic Bézier [P0,P1,P2,P3] first at t1 then at t2,
    producing three consecutive cubic segments.  Also return the two
    derivative‐samples dS(t1) and dS(t2), unscaled ( = R1−R0 ).

    Parameters
    ----------
    Bx, By : array‐like of length 4
      The control‐point coords [x0,x1,x2,x3], [y0,y1,y2,y3].
    t1, t2 : floats
      0 < t1 < t2 < 1

    Returns
    -------
    (B1x, B1y,
     B2x, B2y,
     B3x, B3y,
     dS1x, dS1y,
     dS2x, dS2y)

    - B1x, B1y : (4,) arrays for segment 1: P0→S(t1)
    - B2x, B2y : (4,) arrays for segment 2: S(t1)→S(t2)
    - B3x, B3y : (4,) arrays for segment 3: S(t2)→P3
    - dS1 = (dS1x,dS1y) = R1−R0 at t1
    - dS2 = (dS2x,dS2y) = R1−R0 at t2
    """
    if not (0 < t1 < t2 < 1):
        raise ValueError("Require 0 < t1 < t2 < 1")

    # 1) de Casteljau at t1 on the *original* curve
    Q0_1, Q1_1, Q2_1, R0_1, R1_1, S1 = _de_casteljau(Bx, By, t1)
    dS1x, dS1y = R1_1[0] - R0_1[0], R1_1[1] - R0_1[1]

    # Build first sub‐curve control points: [P0, Q0_1, R0_1, S1]
    B1x = np.array([Bx[0], Q0_1[0], R0_1[0], S1[0]])
    B1y = np.array([By[0], Q0_1[1], R0_1[1], S1[1]])

    # The "right remainder" after t1 is a cubic with CPs [S1, R1_1, Q2_1, P3].
    Bx_r = np.array([S1[0], R1_1[0], Q2_1[0], Bx[3]])
    By_r = np.array([S1[1], R1_1[1], Q2_1[1], By[3]])

    # 2) We want to split that remainder at the *global* t2,
    #    but in the local parameter of the remainder that is
    #    t′ = (t2 − t1)/(1 − t1).
    t_prime = (t2 - t1) / (1 - t1)

    # de Casteljau on the remainder at t′
    Q0_2, Q1_2, Q2_2, R0_2, R1_2, S2 = _de_casteljau(Bx_r, By_r, t_prime)
    # Build second & third segments in global coords:
    B2x = np.array([Bx_r[0], Q0_2[0], R0_2[0], S2[0]])
    B2y = np.array([By_r[0], Q0_2[1], R0_2[1], S2[1]])

    B3x = np.array([S2[0], R1_2[0], Q2_2[0], Bx_r[3]])
    B3y = np.array([S2[1], R1_2[1], Q2_2[1], By_r[3]])

    # 3) Finally compute the derivative‐sample at the *global* t2
    #    directly from the original curve:
    _, _, _, R0t2, R1t2, _ = _de_casteljau(Bx, By, t2)
    dS2x, dS2y = R1t2[0] - R0t2[0], R1t2[1] - R0t2[1]

    return B1x, B1y, B2x, B2y, B3x, B3y, dS1x, dS1y, dS2x, dS2y


def parse_arguments():
    parser = argparse.ArgumentParser(description="Curve Fitting with Bezier")
    parser.add_argument("--read_points", type=bool, default=True, help="Whether to read points from file or generate new ones.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps for the fitting process.")
    parser.add_argument("--use_T_orig", type=bool, default=True, help="Whether to use the original T values.")
    parser.add_argument("--path_to_bezier_points", type=str, default="bezier_points", help="Path to the Bezier points file.")
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
    points = plt.ginput(4, timeout=60)  # 60 segundos
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


def derivative(t, Bx, By):
    d1 = np.array([
        [-3, 3, 0, 0],
        [ 0,-3, 3, 0],
        [ 0, 0,-3, 3]
    ])

    d1_Bx = d1 @ Bx.transpose()
    d1_By = d1 @ By.transpose()

    T_quadratico = construct_quadratic_matrix_T(t)

    X2 = d1_Bx @ T_quadratico.transpose()
    Y2 = d1_By @ T_quadratico.transpose()

    return np.concatenate((X2, Y2))


def compute_step_newton1(T, Bx, By):
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
    f = (X1 - A1) * X2 + (Y1 - B1) * Y2
    # f'(t)
    d1_f = X2**2 + Y2**2 + (X1 - A1) * X3 + (Y1 - B1) * Y3

    # t <- t - f(t)/f'(t)
    T -= f / d1_f

    T = (T - np.min(T)) / (np.max(T) - np.min(T))

    return T

def compute_step_newton2(T, Bx, By):
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
    f = (X1 - A2) * X2 + (Y1 - B2) * Y2
    # f'(t)
    d1_f = X2**2 + Y2**2 + (X1 - A2) * X3 + (Y1 - B2) * Y3

    # t <- t - f(t)/f'(t)
    T -= f / d1_f

    T = (T - np.min(T)) / (np.max(T) - np.min(T))

    return T

def compute_step_newton3(T, Bx, By):
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
    f = (X1 - A3) * X2 + (Y1 - B3) * Y2
    # f'(t)
    d1_f = X2**2 + Y2**2 + (X1 - A3) * X3 + (Y1 - B3) * Y3

    # t <- t - f(t)/f'(t)
    T -= f / d1_f

    T = (T - np.min(T)) / (np.max(T) - np.min(T))

    return T


def update_T1(T, Bx, By):

    error_old = 1
    error = 0

    while abs(error - error_old) > 10**(-7):
        T_old = T
        T = compute_step_newton1(T, Bx, By)
        error_old = error
        error = np.sum((T - T_old)**2)

    return T

def update_T2(T, Bx, By):

    error_old = 1
    error = 0

    while abs(error - error_old) > 10**(-7):
        T_old = T
        T = compute_step_newton2(T, Bx, By)
        error_old = error
        error = np.sum((T - T_old)**2)

    return T

def update_T3(T, Bx, By):

    error_old = 1
    error = 0

    while abs(error - error_old) > 10**(-7):
        T_old = T
        T = compute_step_newton3(T, Bx, By)
        error_old = error
        error = np.sum((T - T_old)**2)

    return T



def solve_linear_regression_fixed_points_and_gradient(t, X_data, Y_data, points, gradient_x, gradient_y, position_gradient):
    initial_point = points[0,:]
    final_point = points[-1,:]


    # Solve
    if position_gradient == 'left' or position_gradient == 'right':
        res = fit_bezier_tangent(t, X_data, Y_data, initial_point[0], final_point[0],
                                 initial_point[1], final_point[1], gradient_x, gradient_y,
                                 position_gradient)
    elif position_gradient == 'both':
        initial_guess = [0, 0]
        res = fit_bezier_tangents_simultaneous(t, X_data, Y_data, initial_point[0], final_point[0],
                                               initial_point[1], final_point[1], gradient_x, gradient_y)
    else:
        ...

    if position_gradient == 'left':
        B1 = [res[0], res[1]]
        B2 = final_point - (res[2] * np.array([gradient_x, gradient_y])) / 3
    elif position_gradient == 'right':
        B1 = initial_point + (res[2] * np.array([gradient_x, gradient_y])) / 3
        B2 = [res[0], res[1]]
    elif position_gradient == 'both':
        B1 = initial_point + (res[1] * np.array([gradient_x[1], gradient_y[1]])) / 3
        B2 = final_point - (res[0] * np.array([gradient_x[0], gradient_y[0]])) / 3
    else:
        ...

    final_coefficients = np.vstack([initial_point, B1, B2, final_point])

    return final_coefficients

# args = parse_arguments()

read_points = True
num_steps = 10
use_T_orig = True
path_to_bezier_points = "/content/bezier_points.npy"

if path_to_bezier_points is None:
    path_to_bezier_points = "./bezier_points.npy"
elif not path_to_bezier_points.endswith(".npy"):
    path_to_bezier_points += ".npy"

if read_points:

    if not os.path.exists(path_to_bezier_points):
        bezier_points = get_bezier_points()
        save_bezier_points(bezier_points, path_to_bezier_points)
    else:
        #bezier_points = load_bezier_points(path_to_bezier_points)
        bezier_points = np.array([[0.66, 0.88], [0.92, 4.66], [4.38, 5.88], [7.58, 3.3]])
        print(bezier_points)

    # Split the points into X and Y vectors:
    Bx = bezier_points[:, 0]
    By = bezier_points[:, 1]

    # Subdivide at t1=1/3 and t2=2/3:
    B1x, B1y, B2x, B2y, B3x, B3y, dS1x, dS1y, dS2x, dS2y = subdivide_cubic_bezier_at_two_ts(Bx, By, t1=1/3, t2=2/3)

    # stack them back for any ground‐truth checks:
    B_gt  = np.vstack([Bx,  By]).T
    B_gt1 = np.vstack([B1x, B1y]).T
    B_gt2 = np.vstack([B2x, B2y]).T
    B_gt3 = np.vstack([B3x, B3y]).T

    # Sample each of the three sub‐curves for plotting:
    A1, B1, T_orig = extract_points_cubix(B1x, B1y, n=50)
    A2, B2, T_orig = extract_points_cubix(B2x, B2y, n=50)
    A3, B3, T_orig = extract_points_cubix(B3x, B3y, n=50)

    # Compute the two split‐points on the curve:
    # S0 = point at t=1/3 is the last pt of segment 1
    # S1 = point at t=2/3 is the last pt of segment 2
    S0 = np.array([B1x[-1], B1y[-1]])
    S1 = np.array([B2x[-1], B2y[-1]])

    # -------------------------------------------------------------------
    # Now do the plotting
    # -------------------------------------------------------------------
    plt.figure(figsize=(8,6))

    # – The three Bézier segments
    plt.plot(A1, B1, 'b-', label='Left Sub‐Curve (0 → 1/3)')
    plt.plot(A2, B2, 'r-', label='Middle Sub‐Curve (1/3 → 2/3)')
    plt.plot(A3, B3, 'g-', label='Right Sub‐Curve (2/3 → 1)')

    # – Arrow at t=1/3 using (dS1x, dS1y)
    plt.arrow(
        S0[0], S0[1],
        dS1x, dS1y,
        head_width=0.05, head_length=0.1,
        fc='k', ec='k',
        alpha=0.7,
        length_includes_head=True
    )

    # – Arrow at t=2/3 using (dS2x, dS2y)
    plt.arrow(
        S1[0], S1[1],
        dS2x, dS2y,
        head_width=0.05, head_length=0.1,
        fc='m', ec='m',
        alpha=0.7,
        length_includes_head=True
    )

    # – Control‐polygons
    plt.plot(Bx,   By,   'ko--', label='Original CP')
    plt.plot(B1x,  B1y,  'bo--', label='Left CP (0→1/3)')
    plt.plot(B2x,  B2y,  'ro--', label='Mid CP (1/3→2/3)')
    plt.plot(B3x,  B3y,  'go--', label='Right CP (2/3→1)')

    # – Labels, grid, aspect
    plt.title("Cubic Bézier subdivided at t=1/3 & t=2/3", alpha=0.6)
    plt.xlabel("x", alpha=0.6)
    plt.ylabel("y", alpha=0.6)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # – Semi‐transparent legend
    leg = plt.legend(framealpha=0.3, edgecolor='gray')
    for txt in leg.get_texts():
        txt.set_alpha(0.7)

    plt.show()



else:

    # Pegar os pontos
    with open('example.eps', 'r') as f:
        eps = f.read()

    X_pre, Y_pre = extract_points(eps)

    # Fixar 'n' para testar
    n = 30

    X = X_pre[2:n + 3]
    Y = Y_pre[2:n + 3]




# --- Código inicial ---
if use_T_orig:
    T1 = T_orig
    T2 = T_orig
    T3 = T_orig
    num_steps = 1
else:
    T1 = initialize_T(A1, B1)
    T2 = initialize_T(A2, B2)
    T3 = initialize_T(A3, B3)
    num_steps = 100

points1 = np.vstack([A1, B1]).T
points2 = np.vstack([A2, B2]).T
points3 = np.vstack([A3, B3]).T
errors = []
fig, ax = plt.subplots(figsize=(8, 6))

# Salvar todos os pontos para o vídeo depois
saved_points_x1 = []
saved_points_y1 = []
saved_points_x2 = []
saved_points_y2 = []
saved_points_x3 = []
saved_points_y3 = []

k = 0
c = 0

plt.ion()  # modo interativo

while True:  # loop infinito

    if not use_T_orig:
        T1 = initialize_T(A1, B1)
        T2 = initialize_T(A2, B2)
        T3 = initialize_T(A3, B3)

    k = k + 1
    for i in range(num_steps):
        matrix_t1 = construct_cubical_matrix_T(T1)
        matrix_t2 = construct_cubical_matrix_T(T2)
        matrix_t3 = construct_cubical_matrix_T(T3)

        extracted_coefficients1 = solve_linear_regression_fixed_points_and_gradient(T1, A1, B1, points1,
                                                                                     gradient_x = dS1x,
                                                                                     gradient_y = dS1y,
                                                                                     position_gradient='left')
        extracted_coefficients2 = solve_linear_regression_fixed_points_and_gradient(T2, A2, B2, points2,
                                                                                     gradient_x = [dS2x, dS1x],
                                                                                     gradient_y = [dS2y, dS1y],
                                                                                     position_gradient='both')
        extracted_coefficients3 = solve_linear_regression_fixed_points_and_gradient(T3, A3, B3, points3,
                                                                                     gradient_x = dS2x,
                                                                                     gradient_y = dS2y,
                                                                                     position_gradient='right')


        T1 = update_T1(T1, extracted_coefficients1[:, 0], extracted_coefficients1[:, 1])
        T2 = update_T2(T2, extracted_coefficients2[:, 0], extracted_coefficients2[:, 1])
        T3 = update_T3(T3, extracted_coefficients3[:, 0], extracted_coefficients3[:, 1])

        points_extracted_from_cubix1 = extract_points_cubix(extracted_coefficients1[:, 0], extracted_coefficients1[:, 1], n=50)
        points_extracted_from_cubix2 = extract_points_cubix(extracted_coefficients2[:, 0], extracted_coefficients2[:, 1], n=50)
        points_extracted_from_cubix3 = extract_points_cubix(extracted_coefficients3[:, 0], extracted_coefficients3[:, 1], n=50)

        print("Derivada verdadeira no ponto 1/3: ", dS1x, dS1y)
        dc1_1 = derivative(1, extracted_coefficients1[:, 0], extracted_coefficients1[:, 1])
        print("Derivada da curva 1 no ponto 1: ", dc1_1[0], dc1_1[1])
        dc2_0 = derivative(0, extracted_coefficients2[:, 0], extracted_coefficients2[:, 1])
        print("Derivada da curva 2 no ponto 0: ", dc2_0[0], dc2_0[1])


        print("Derivada verdadeira no ponto 2/3: ", dS2x, dS2y)
        dc2_1 = derivative(1, extracted_coefficients2[:, 0], extracted_coefficients2[:, 1])
        print("Derivada da curva 2 no ponto 1: ", dc2_1[0], dc2_1[1])
        dc3_0 = derivative(0, extracted_coefficients3[:, 0], extracted_coefficients3[:, 1])
        print("Derivada da curva 3 no ponto 0: ", dc3_0[0], dc3_0[1])


        saved_points_x1.append(points_extracted_from_cubix1[0])
        saved_points_y1.append(points_extracted_from_cubix1[1])
        saved_points_x2.append(points_extracted_from_cubix2[0])
        saved_points_y2.append(points_extracted_from_cubix2[1])
        saved_points_x3.append(points_extracted_from_cubix3[0])
        saved_points_y3.append(points_extracted_from_cubix3[1])

        ax.clear()
        ax.scatter(A1, B1, label='Dados Originais 1')
        ax.scatter(A2, B2, label='Dados Originais 2')
        ax.scatter(A3, B3, label='Dados Originais 3')

        ax.plot(points_extracted_from_cubix1[0], points_extracted_from_cubix1[1], 'r')
        ax.plot(points_extracted_from_cubix2[0], points_extracted_from_cubix2[1], 'g')
        ax.plot(points_extracted_from_cubix3[0], points_extracted_from_cubix3[1], 'b')
        ax.legend()
        ax.set_title(f'Iteração {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        '''
        interx1, intery1 = points_extracted_from_cubix2[0][0], points_extracted_from_cubix2[1][0]
        interx2, intery2 = points_extracted_from_cubix2[0][-1], points_extracted_from_cubix2[1][-1]

        # zoom
        dx = 0.5
        dy = 0.5

        # 3) set the limits on your existing axes
        ax.set_xlim(interx1 - dx, interx1 + dx)
        ax.set_ylim(intery1 - dy, intery1 + dy)

        plt.draw()

        ax.set_xlim(interx2 - dx, interx2 + dx)
        ax.set_ylim(intery2 - dy, intery2 + dy)

        plt.draw()

        plt.pause(0.05)
        '''

        if read_points:
            error1 = np.linalg.norm(B_gt1 - extracted_coefficients1)
            error2 = np.linalg.norm(B_gt2 - extracted_coefficients2)
            B_gt_all = np.vstack([B_gt1, B_gt2])
            B_fit_all = np.vstack([extracted_coefficients1, extracted_coefficients2])
            error = np.linalg.norm(B_gt_all - B_fit_all)
            errors.append(error)
            print(f'Iteração {i+1}: Error = {error:.4f}')

        if not plt.get_fignums():
            print("Janela fechada. Saindo do loop.")
            c = 1
            break

    if c == 1:
        break

    if k == 1:
        plt.figure()
        plt.plot(errors)
        plt.xlabel('Passo')
        plt.ylabel('Erro')
        plt.title('Erro em função do passo')
        plt.grid(True)
        plt.show()

plt.ioff()  # desliga modo interativo depois

# --- Agora salva o vídeo com os pontos gravados ---
print("Salvando o vídeo...")

fig2, ax2 = plt.subplots(figsize=(8, 6))

