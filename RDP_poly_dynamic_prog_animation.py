from typing_extensions import final
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import argparse
import matplotlib.animation as animation
from extract_tangents import sliding_window

import scipy.optimize as opt


def X_LR(t, alpha_l, alpha_r, x_0, x_m, vx):
    B1x = x_0 + (alpha_r * vx[1]) / 3
    B2x = x_m - (alpha_r * vx[0]) / 3
    return x_0 * (1-t)**3 + B1x * 3 * t * (1-t)**2 + B2x * 3 * t**2 * (1-t) + x_m * t**3

def Y_LR(t, alpha_l, alpha_r, y_0, y_m, vy):
    B1y = y_0 + (alpha_r * vy[1]) / 3
    B2y = y_m - (alpha_l * vy[0]) / 3
    return y_0 * (1-t)**3 + B1y * 3 * t * (1-t)**2 + B2y * 3 * t**2 * (1-t) + y_m * t**3

def P_LR(t, alpha_l, alpha_r, p_0, p_m, gradient_left, gradient_right):
    B1y = p_0 + (alpha_r * gradient_right) / 3
    B2y = p_m - (alpha_l * gradient_left) / 3

    t = np.array(t)

    if t.ndim > 0:
        p_0 = p_0[:, np.newaxis]
        B1y = B1y[:, np.newaxis]
        B2y = B2y[:, np.newaxis]
        p_m = p_m[:, np.newaxis]

    return (p_0 * (1-t)**3 + B1y * 3 * t * (1-t)**2 + B2y * 3 * t**2 * (1-t) + p_m * t**3).transpose()


def error_function_LR(params, t, P, gradient_left, gradient_right):
    p_0, p_m = P[0,:], P[-1,:]
    alpha_l, alpha_r = params
    P_calculated = P_LR(t, alpha_l, alpha_r, p_0, p_m, gradient_left, gradient_right)

    return np.sum((P_calculated - P)**2)


def solve_linear_regression_fixed_points_and_gradient(t, P, gradient_left, gradient_right):
    initial_point = P[0,:]
    final_point = P[-1,:]

    initial_guess = [0, 0]
    alpha_l, alpha_r = opt.minimize(error_function_LR, initial_guess, args=(t, P, gradient_left, gradient_right)).x

    # Pode estar errado, cuidado
    B1 = initial_point + (alpha_r * gradient_right) / 3
    B2 = final_point - (alpha_l * gradient_left) / 3

    final_coefficients = np.vstack([initial_point, B1, B2, final_point])

    return final_coefficients


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


def extract_points_cubix(bezier, n = 50):
    t = np.linspace(0, 1, n)

    bezier = bezier[:, np.newaxis]
    t = t[:, np.newaxis]

    # Calcular x e y usando o polinomio
    p = bezier[0]*(1-t)**3 + bezier[1]*(3*((1-t)**2)*t)  + bezier[2] * (3*(1-t)*(t**2))  + bezier[3] * (t**3)

    return p, t

def solve_linear_regression_fixed_points(matrix_t, P):
    # Pegar pontos referentes ao menor e maior t
    min_t_index = np.argmin(matrix_t[:,3])
    max_t_index = np.argmax(matrix_t[:,3])

    initial_point = P[min_t_index,:]
    final_point = P[max_t_index,:]

    factor_initial_point = matrix_t[:,0:1] * initial_point
    factor_final_point = matrix_t[:,-1:] * final_point

    matrix_t_no_endpoints = matrix_t[:, 1:-1]

    points_translated = P - factor_final_point - factor_initial_point
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

def solve_linear_regression(matrix_t, P):

    transform_y = matrix_t.T @ P
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

def initialize_T(P):
    T = [0.0]
    s = 0.0

    n = len(P)

    for i in range(1, n):
        s += np.sqrt(np.sum((P[i] - P[i-1])**2))
        T.append(s)

    # T final
    if n != 1:
        T = np.array(T) / s
    
    return T

def compute_step_newton(T, bezier, P):
    T = np.array(T)

    d1 = np.array([
        [-3, 3, 0, 0],
        [ 0,-3, 3, 0],
        [ 0, 0,-3, 3]
    ])

    d1_bezier = d1 @ bezier

    # Segunda derivada
    d2 = np.array([
        [  6,-12,  6,  0],
        [  0,  6,-12,  6],
    ])

    d2_bezier = d2 @ bezier

    # T_cúbico
    T_cubico = construct_cubical_matrix_T(T)

    P1 = (bezier.transpose() @ T_cubico.transpose()).transpose()

    # T_quadratico
    T_quadratico = construct_quadratic_matrix_T(T)

    P2 = (d1_bezier.transpose() @ T_quadratico.transpose()).transpose()

    # T_linear
    T_linear = construct_linear_matrix_T(T)

    P3 = (d2_bezier.transpose() @ T_linear.transpose()).transpose()

    # f(t)
    f = np.sum((P1 - P) * P2)

    # f'(t)
    d1_f = np.sum(P2**2 + (P1 - P) * P3)
    if d1_f == 0:
        d1_f = 10**(-7)

    # t <- t - f(t)/f'(t)
    T -= f / d1_f

    if np.max(T) != np.min(T):
        T = (T - np.min(T)) / (np.max(T) - np.min(T))
    else:
        T = (T - np.min(T))

    return T


def update_T(T, bezier, P):

    error_old = 1
    error = 0
    inter = 0

    while (abs(error - error_old) > 10**(-7)) or (inter == 24):
        T_old = T
        T = compute_step_newton(T, bezier, P)
        error_old = error
        error = np.sum((T - T_old)**2)
        inter += 1

    return T

def visualize_fitting(X, Y,T, Bx, By, use_T_orig, num_steps, read_points, B_gt = None):

    P = np.vstack([X, Y]).T
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
            extracted_coefficients = solve_linear_regression(matrix_t, P)

            T = update_T(T, extracted_coefficients, P)

            points_extracted_from_cubix = extract_points_cubix(extracted_coefficients, n=50)

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


def fit_directly_bezier(P, num_steps, gradient_left, gradient_right):

    T = initialize_T(P)

    for i in range(num_steps):
        extracted_coefficients = solve_linear_regression_fixed_points_and_gradient(T, P, gradient_left, gradient_right)
        T = update_T(T, extracted_coefficients, P)

    return extracted_coefficients

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

def point_line_distance(pt, a, b):
    """Perpendicular distance from pt to the line segment a→b."""
    # vector form
    v = b - a
    w = pt - a
    # projection t of w onto v, clamped to [0,1]
    t = np.dot(w, v) / np.dot(v, v)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * v
    return np.linalg.norm(pt - proj)



def dp_douglas_peucker(P, epsilon, method="max"):
    """
    DP-based Ramer–Douglas–Peucker with vectorized inner loops.

    Parameters:
        P (numpy.ndarray): Array of shape (n,2).
        epsilon (float): Distance threshold.
        method (str): "max" or "middle".

    Returns:
        List of points (as numpy arrays) after simplification.
    """
    pts = np.asarray(P, dtype=float)
    n = len(pts)
    if n < 2:
        return pts.tolist()

    # 1) Build error matrix err[i,j]
    err = np.zeros((n, n), dtype=float)

    for i in range(n - 1):
        start = pts[i]
        # vectors from P_i to all later points
        rel = pts[i+1:] - start           # shape (m,2), m = n-i-1
        norms = np.linalg.norm(rel, axis=1)
        norms[norms == 0] = 1.0           # avoid zero‐divide

        # for each j>i+1 compute err[i,j] using vectorized cross-products
        # offset runs from 1..m-1  corresponding to j = i+1+offset
        for offset in range(1, rel.shape[0]):
            v = rel[offset]               # P_j – P_i
            L = norms[offset]             # ||v||

            if method == "max":
                # interior vectors = rel[1:offset]
                interior = rel[1:offset]   # shape (offset-1,2)
                # cross ‖r × v‖ = |r_x * v_y – r_y * v_x|
                cross = np.abs(interior[:,0]*v[1] - interior[:,1]*v[0])
                err[i, i+1+offset] = cross.max() / L if interior.size else 0.0

            elif method == "middle":
                mid_idx = 1 + (offset - 1)//2
                r_mid = rel[mid_idx]
                err[i, i+1+offset] = abs(r_mid[0]*v[1] - r_mid[1]*v[0]) / L

            else:
                raise ValueError("method must be 'max' or 'middle'")

    # 2) DP to find minimal cuts
    dp = [float('inf')] * n
    prev = [-1] * n
    dp[0] = 0

    for j in range(1, n):
        for i in range(j):
            if j == i + 1 or err[i, j] <= epsilon:
                cost = dp[i] + 1
                if cost < dp[j]:
                    dp[j] = cost
                    prev[j] = i

    # 3) Backtrack to recover kept indices
    indices = []
    cur = n - 1
    while cur != 0:
        indices.append(cur)
        cur = prev[cur]
    indices.append(0)
    indices.reverse()

    # 4) Return list of points
    return [pts[idx] for idx in indices], indices


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


        P, T_orig = extract_points_cubix(bezier_points, n=100) #extract_points(eps)

    else:

        # Pegar os pontos
        with open('3.eps', 'r') as f:
            eps = f.read()

        X, Y = extract_points(eps)

        P = np.column_stack((X, Y))


    # --- Código inicial ---
    if use_T_orig:
        T = T_orig
        num_steps = 1
    else:
        T = initialize_T(P)

    if choose_knots:

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

        plt.axis('equal')
        plt.show()

        fig, ax = plt.subplots()

        def update(frame):
            # Limpa o gráfico anterior
            ax.clear()
            
            epsilon = (frame+1)  # Normaliza o frame para valores menores
            knots, indices = dp_douglas_peucker(P, epsilon)
            knots = np.array(knots)

            points_from_knots_bezier = []

            steps = 25
            num_points = 50

            for i in range(len(indices)-1):
                idx = indices[i]
                next_idx = indices[i+1]
                curve_bezier = fit_directly_bezier(P[idx:next_idx+1], steps, tangent_points[idx], tangent_points[next_idx])
                curve_points, _ = extract_points_cubix(curve_bezier, num_points)
                points_from_knots_bezier.append(curve_points)
            
            # caso final tbm
            idx = indices[-1]
            next_idx = indices[0]
            last_points = np.concatenate((P[idx:], P[:next_idx]))
            curve_bezier = fit_directly_bezier(last_points, steps, tangent_points[idx], tangent_points[next_idx])
            curve_points, _ = extract_points_cubix(curve_bezier, num_points)
            points_from_knots_bezier.append(curve_points)

            # Plotagem
            ax.plot(X, Y, label='Dados Originais')
            ax.set_title(f'ε = {epsilon:.2f}')
            ax.scatter(knots[:,0], knots[:,1], label='Pontos de controle',color='orange')
            ax.plot(P[:,0], P[:,1], label='Pontos da curva',color='red')
            ax.legend()
            plt.axis('equal')
            print(epsilon)

        # Crie a animação
        anim = animation.FuncAnimation(fig, update, frames=range(300), interval=50)

        # Salvar como GIF
        anim.save('RDP_epsilon1.gif', writer='pillow')

        #plt.show()


    else:

        
        extracted_coefficients = extracted_coefficients_list[0]
        P, _ = extract_points_cubix(extracted_coefficients, n=50)


        


if "__main__" == __name__:

    main()
