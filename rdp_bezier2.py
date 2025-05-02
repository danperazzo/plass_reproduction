from typing_extensions import final
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import argparse
import matplotlib.animation as animation
from extract_tangents import sliding_window

def parse_arguments():
    parser = argparse.ArgumentParser(description="Curve Fitting with Bezier")
    parser.add_argument("--read_points", type=bool, default=True, help="Whether to read points from file or generate new ones.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps for the fitting process.")
    parser.add_argument("--use_T_orig", type=bool, default=False, help="Whether to use the original T values.")
    parser.add_argument("--path_to_bezier_points", type=str, default="bezier_points", help="Path to the Bezier points file.")
    parser.add_argument("--visualize_evolution", type=bool, default=False, help="Whether to visualize the evolution of the fitting process.")
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

def compute_step_newton(T, Bx, By, X, Y):
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


def update_T(T, Bx, By, X, Y):

    error_old = 1
    error = 0
    iteration = 0

    while abs(error - error_old) > 10**(-7):
        T_old = T
        T = compute_step_newton(T, Bx, By, X, Y)
        error_old = error
        error = np.sum((T - T_old)**2)
        iteration += 1
        if iteration > 50:
            break

    return T

def visualize_fitting(X, Y,T, Bx, By, use_T_orig, num_steps, read_points, B_gt = None):

    points = np.vstack([X, Y]).T
    errors = []
    fig, ax = plt.subplots(figsize=(8, 6))

    # Salvar todos os pontos para o vídeo depois
    saved_points_x = []
    saved_points_y = []

    k = 0
    c = 0

    plt.ion()  # modo interativo

    while True:  # loop infinito

        if not use_T_orig:
            T = initialize_T(X, Y)

        k = k + 1
        for i in range(num_steps):
            matrix_t = construct_cubical_matrix_T(T)
            extracted_coefficients = solve_linear_regression(matrix_t, points)

            T = update_T(T, extracted_coefficients[:, 0], extracted_coefficients[:, 1], X, Y)

            points_extracted_from_cubix = extract_points_cubix(extracted_coefficients[:, 0], extracted_coefficients[:, 1], n=50)

            saved_points_x.append(points_extracted_from_cubix[0])
            saved_points_y.append(points_extracted_from_cubix[1])

            ax.clear()
            ax.scatter(X, Y, label='Dados Originais')
            ax.plot(points_extracted_from_cubix[0], points_extracted_from_cubix[1], 'r', label=f'Iteração {i+1}')
            ax.legend()
            ax.set_title(f'Iteração {i+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.pause(1)

            if read_points:
                error = np.linalg.norm(B_gt - extracted_coefficients)
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

    def update(frame_idx):
        ax2.clear()
        ax2.scatter(X, Y, label='Dados Originais')
        ax2.plot(saved_points_x[frame_idx], saved_points_y[frame_idx], 'r', label=f'Iteração {frame_idx+1}')
        ax2.legend()
        ax2.set_title(f'Iteração {frame_idx+1}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_xlim(np.min(X)-1, np.max(X)+1)
        ax2.set_ylim(np.min(Y)-1, np.max(Y)+1)

    ani = animation.FuncAnimation(fig2, update, frames= num_steps, interval=300)

    ani.save('evolucao.mp4', writer='ffmpeg', fps=5)
    print("Vídeo salvo como evolucao.mp4 🎬")


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


def solve_linear_regression_fixed_points_and_gradient(t, points, gradient_x, gradient_y, position_gradient):
    initial_point = points[0,:]
    final_point = points[-1,:]


    # Solve
    if position_gradient == 'left' or position_gradient == 'right':
        res = fit_bezier_tangent(t, points[:, 0], points[:, 1], initial_point[0], final_point[0],
                                 initial_point[1], final_point[1], gradient_x, gradient_y,
                                 position_gradient)
    elif position_gradient == 'both':
        initial_guess = [0, 0]
        res = fit_bezier_tangents_simultaneous(t, points[:, 0], points[:, 1], initial_point[0], final_point[0],
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


def fit_directly_bezier(X, Y, tangent_left, tangent_right, num_steps):

    points = np.vstack([X, Y]).T
    T = initialize_T(X, Y)
    grad_x = [tangent_left[0], tangent_right[0]]
    grad_y = [tangent_left[1], tangent_right[1]]

    for i in range(num_steps):
        extracted_coefficients = solve_linear_regression_fixed_points_and_gradient(T, points, grad_x, grad_y, "both")
        T = update_T(T, extracted_coefficients[:, 0], extracted_coefficients[:, 1], X, Y)

    return extracted_coefficients, T

def extract_tangent_of_bezier(Bx, By, time):
    """
    Extract the tangent of a Bezier curve at a given time.

    Parameters:
        Bx (numpy.ndarray): X coordinates of the Bezier control points.
        By (numpy.ndarray): Y coordinates of the Bezier control points.
        time (float): The time parameter at which to extract the tangent.

    Returns:
        tuple: Tangent vector (dx, dy) at the specified time.
    """
    dx = 3 * ((1 - time) ** 2) * (Bx[1] - Bx[0]) + 6 * (1 - time) * time * (Bx[2] - Bx[1]) + 3 * (time ** 2) * (Bx[3] - Bx[2])
    dy = 3 * ((1 - time) ** 2) * (By[1] - By[0]) + 6 * (1 - time) * time * (By[2] - By[1]) + 3 * (time ** 2) * (By[3] - By[2])
    return dx, dy


def bezier_dist(points, T, coefficients):
    matrix_t = construct_cubical_matrix_T(T)
    bezier_points = matrix_t @ coefficients
    distances = np.linalg.norm(points - bezier_points, axis=1)
    return distances


def rdp_bezier(points, tangents, epsilon, method="max"):
    """
    Ramer-Douglas-Peucker algorithm adapted for Bezier curve simplification.

    Parameters:
        points (numpy.ndarray): A 2D array of points (shape: [n, 2]).
        T (numpy.ndarray): Parameter values corresponding to points (shape: [n,]).
                           Needed if fit_directly_bezier uses it.
        epsilon (float): The distance threshold for simplification.
        method (str): Method to choose the subdivision point.
                      "max" for maximum error, "middle" for the middle point.

    Returns:
        numpy.ndarray: A simplified array of points (shape: [m, 2]), where m <= n.
                       Returns points as a numpy array for consistency.
    """
    n = len(points)
    start, end = points[0], points[-1]

    if n <= 2:
        return points

    coefficients, T_new = fit_directly_bezier(points[:, 0], points[:, 1], tangents[0], tangents[-1], 10) # Example params

    distances = bezier_dist(points, T_new, coefficients)

    max_dist = -1.0 # Initialize
    max_idx = -1    # Initialize

    if method == "max":
        if n > 2: # Only possible if there are intermediate points
            intermediate_distances = distances[1:-1]
            if len(intermediate_distances) > 0:
                max_dist = np.max(intermediate_distances)
                max_idx_intermediate = np.argmax(intermediate_distances)
                max_idx = max_idx_intermediate + 1

    elif method == "middle":
        # Choose the middle point index (safe for n > 2)
        max_idx = n // 2
        max_dist = distances[max_idx] # Distance at the middle point
    else:
        raise ValueError("Invalid method. Use 'max' or 'middle'.")

    if max_idx != -1 and max_dist > epsilon:
        left_points = points[:max_idx + 1]
        left_tangents = tangents[:max_idx + 1]
        recursive_left = rdp_bezier(left_points, left_tangents, epsilon, method)

        right_points = points[max_idx:]
        right_tangents = tangents[max_idx:]
        recursive_right = rdp_bezier(right_points, right_tangents, epsilon, method)

        return np.vstack((recursive_left[:-1], recursive_right))
    else:
        return np.array([start, end])

def main():

    read_points = False
    num_steps = 1
    use_T_orig = False
    path_to_bezier_points = False
    choose_knots = True


    if read_points:

        if not os.path.exists(path_to_bezier_points):
            bezier_points = get_bezier_points()
            save_bezier_points(bezier_points, path_to_bezier_points)
        else:
            bezier_points = load_bezier_points(path_to_bezier_points)

        Bx = bezier_points[:, 0]
        By = bezier_points[:, 1]

        B_gt = np.vstack([Bx, By]).T
        X, Y, T_orig = extract_points_cubix(Bx, By, n=100) #extract_points(eps)

    else:

        # Pegar os pontos
        points = np.loadtxt('/content/3.txt', delimiter=' ')  # Adjust delimiter if needed
        X_pre = points[:, 0]
        Y_pre = points[:, 1]

        # Fixar 'n' para testar
        n = 100

        desloc = 15

        X = X_pre#[2 + desloc:n + 3 + desloc]
        Y = Y_pre#[2 + desloc:n + 3 + desloc]


    # --- Código inicial ---
    if use_T_orig:
        T = T_orig
        num_steps = 1
    else:
        T = initialize_T(X, Y)

    if choose_knots:
        P = np.column_stack((X, Y))

        extracted_coefficients_list, tangent_points = sliding_window(X, Y, T, use_T_orig, num_steps, window_size=10)



        fig, ax = plt.subplots()
        ax.plot(X, Y, label='Dados Originais')



        # Add tangent vectors as arrows
        for i, (x, y) in enumerate(zip(X, Y)):
            if i < len(tangent_points):  # Ensure we have a tangent for this point
                dx, dy = tangent_points[i]
                ax.quiver(x, y, dy, -dx, angles='xy', scale_units='xy', scale=1, color='blue', label='Tangente' if i == 0 else "")

        ax.legend()
        ax.set_title('Curva Ajustada')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

        knots = rdp_bezier(P, tangent_points, 50.0, "max")
        knots = np.array(knots)

        fig, ax = plt.subplots()

        ax.plot(X, Y, label='Dados Originais')

        print(len(knots))

        ax.plot(knots[:,0], knots[:,1], label='Pontos de controle',color='red', marker='o')
        ax.legend()
        plt.show()

    else:

        
        extracted_coefficients = extracted_coefficients_list[0]
        points = extract_points_cubix(Bx, By, n=50)


        


if "__main__" == __name__:

    main()