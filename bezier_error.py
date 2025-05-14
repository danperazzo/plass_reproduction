# Standard library imports
import os
import re
import argparse

# Third-party imports
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.optimize as opt
from typing_extensions import final
from tqdm import tqdm
from rdp_bezier2 import fit_directly_bezier


class BezierCurveFitter:
    """Handles the fitting of Bezier curves to point sets."""

    def __init__(self, P, max_iterations=21):
        self.MAX_ITERATIONS = max_iterations

        self.d1 = np.array([[-3, 3, 0, 0],
                            [ 0,-3, 3, 0],
                            [ 0, 0,-3, 3]])
        self.d2 = np.array([[ 6,-12, 6, 0],
                            [ 0, 6,-12, 6]])

        self.points = P
        self.T = self.initialize_T(P)


    def initialize_T(self, P):
        T = [0.0]
        s = 0.0

        n = len(P)

        for i in range(1, n):
            s += np.linalg.norm(P[i] - P[i-1])
            T.append(s)

        # T final
        if n > 1:
            T = np.array(T) / s
        
        return T
    

    def construct_matrix_T(self, t, n):
        n_degree_matrix = np.vstack([math.comb(n, k) * ((1-t)**(n-k)) * (t**k) for k in range(n+1)]).T
        return n_degree_matrix
    

    def update_T(self, P, T, coefficients):
        error_old = 1
        error = 0
        num_iterations = 0

        while (abs(error - error_old) > 1e-7) and (num_iterations < self.MAX_ITERATIONS):
            T_old = T
            T = self.compute_step_newton(P, T, coefficients)
            error_old = error
            error = np.sum((T - T_old)**2)
            num_iterations += 1

        return T
    

    def compute_step_newton(self, P, T, coefficients):
        
        d1_bezier = self.d1 @ coefficients
        d2_bezier = self.d2 @ coefficients

        T_cubic = self.construct_matrix_T(T, 3)
        T_quad = self.construct_matrix_T(T, 2)
        T_linear = self.construct_matrix_T(T, 1)

        P1 = (coefficients.T @ T_cubic.T).T
        P2 = (d1_bezier.T @ T_quad.T).T
        P3 = (d2_bezier.T @ T_linear.T).T

        f = np.sum((P1 - P) * P2)
        d1_f = np.sum(P2**2 + (P1 - P) * P3)

        epsilon = 1e-7
        if np.abs(d1_f) < epsilon:
            d1_f = epsilon

        T -= f / d1_f

        T_min, T_max = np.min(T), np.max(T)
        if T_max != T_min:
            T = (T - T_min) / (T_max - T_min)
        else:
            T = T - T_min

        return T
    

    def solve_linear_regression(self, P, matrix_t):
        if P.shape[0] == 2:
            return np.array([P[0], [0, 0], [0, 0], P[1]])
        elif P.shape[0] == 3:
            return np.array([P[0], P[1], P[1], P[2]])
        else:
            transformed_target = matrix_t.T @ P
            coefficients = np.linalg.solve(matrix_t.T @ matrix_t, transformed_target)

        return coefficients

    
    def fit_directly_bezier(self, P, num_steps):

        T  = self.initialize_T(P)

        for i in range(num_steps):
            matrix_t = self.construct_matrix_T(T, 3)
            bezier_coefficients = self.solve_linear_regression(P, matrix_t)
            T = self.update_T(P, T, bezier_coefficients)

        return bezier_coefficients
    

    def extract_points_bezier(self, coefficients, n=50):
        T = np.linspace(0, 1, n)
        
        T_cubic = self.construct_matrix_T(T, 3)

        points = T_cubic @ coefficients

        return points, T
    

    def calculate_points_bezier(self, T, coefficients):
        T_cubic = self.construct_matrix_T(T, 3)

        points = T_cubic @ coefficients

        return points, T
    

    def extract_tangent_of_bezier(self, coefficients, T):
        d1_bezier = self.d1 @ coefficients
        
        T_quad = self.construct_matrix_T(T, 2)
        
        tangents = T_quad @ d1_bezier

        return tangents

    
    def sliding_window(self, num_steps, window_size=5):
        n = len(self.points)
        fitted_curves = []
        tangent_points = []

        for i in tqdm(range(n), desc="Fitting Bezier curves"):

            left_interval = max(0, i - window_size)
            right_interval = min(n, i + window_size)
            P_window = self.points[left_interval:right_interval].copy()
            T_window = self.T[left_interval:right_interval].copy()

            if np.max(T_window) != np.min(T_window):
                T_window = (T_window - np.min(T_window)) / (np.max(T_window) - np.min(T_window))
            else:
                T_window = (T_window - np.min(T_window))
    
            T_center = self.T[i]

            fitted_curve = self.fit_directly_bezier(P_window, num_steps)
            fitted_curves.append(fitted_curve)
            tangent_points.append(self.extract_tangent_of_bezier(fitted_curve, T_center)[0])

        tangent_points = np.array(tangent_points)/20
        return fitted_curves, tangent_points
    

    def bezier_dist(self, P, T, coefficients):
        matrix_t = self.construct_matrix_T(T, 3)
        bezier_points = matrix_t @ coefficients
        distances = np.linalg.norm(P - bezier_points, axis=1)
        return distances
    

    def rdp_bezier_error(self, P, tangents, epsilon):
        n = len(P)
        
        error_matrix = np.zeros((n, n), dtype=float)

        test = [True for i in range(n)]
        times = 0

        for i in tqdm(range(n), desc="Creating the error matrix"):
            gradient_left = tangents[i]

            for j in range(i + 2, n):
                # Todas as curvas de bezier de P_i até todos os próximos pontos
                gradient_right = tangents[j]
                coefficients, T_new = fit_directly_bezier(P[i:j+1][:,0], P[i:j+1][:,1], gradient_left, gradient_right, 1)
                distances = self.bezier_dist(P[i:j+1], T_new, coefficients)
                error_matrix[i, j] = distances[1:-1].max() if distances.size > 2 else 0.0

                if error_matrix[i, j] < epsilon and error_matrix[i, j-1] > epsilon:
                    times += 1
                    test[i] = False
                    
                if error_matrix[i, j] > epsilon * 10:
                    error_matrix[i, j:] = error_matrix[i, j]
                    break
        
        if False in test:
            print(f"Final test failed {times} times")

        else:
            print("It worked")
                
        return error_matrix
    

    def dp_rdp_bezier(self, P, tangents, epsilon, error_matrix=None):
        """
        Ramer-Douglas-Peucker algorithm adapted for Bezier curve simplification.

        Parameters:
            P (numpy.ndarray): A 2D array of points (shape: [n, 2]).
            T (numpy.ndarray): Parameter values corresponding to points (shape: [n,]).
                            Needed if fit_directly_bezier uses it.
            epsilon (float): The distance threshold for simplification.
            method (str): Method to choose the subdivision point.
                        "max" for maximum error, "middle" for the middle point.

        Returns:
            numpy.ndarray: A simplified array of points (shape: [m, 2]), where m <= n.
                        Returns points as a numpy array for consistency.
        """
        n = len(P)
        if n < 2:
            return P.tolist()
        
        if error_matrix is None:
            error_matrix = self.rdp_bezier_error(P, tangents, epsilon)
                

        # Dynamic programming for minimal cuts
        dp = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=int)
        dp[0] = 0

        for j in range(1, n):
            for i in range(j):
                if j == i + 1 or error_matrix[i, j] <= epsilon:
                    cost = dp[i] + 1
                    if cost < dp[j]:
                        dp[j] = cost
                        prev[j] = i

        # Backtrack to find indices of kept points
        indices = []
        cur = n - 1
        while cur != 0:
            indices.append(cur)
            cur = prev[cur]
        indices.append(0)
        indices.reverse()

        knots = [P[idx] for idx in indices]
        
        return knots, indices







############### TEST #######################


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


def main():

    read_points = False
    num_steps = 1
    use_T_orig = False
    path_to_bezier_points = False
    choose_knots = True


    # Pegar os pontos
    points = np.loadtxt('3.txt', delimiter=' ')  # Adjust delimiter if needed
    X = points[:, 0]
    Y = points[:, 1]

    P = np.column_stack((X, Y))


    # --- Código inicial ---
    Bezier = BezierCurveFitter(P)

    if choose_knots:

        extracted_coefficients_list, tangent_points = Bezier.sliding_window(num_steps, window_size=5)

        fig, ax = plt.subplots()


        ax.plot(P[:, 0], P[:, 1], label='Dados Originais')


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

        np.random.seed(22)

        #error_matrix = np.load("error_matrix1.npy") #Bezier.rdp_bezier_error(P, tangent_points)

        #np.save("error_matrix1.npy", error_matrix)

        def update(frame):
            ax.clear()

            epsilon = 5 * (frame+1)  # Normaliza o frame para valores menores
            knots, indices = Bezier.dp_rdp_bezier(P, tangent_points, epsilon)
            knots = np.array(knots)

            points_from_knots_bezier = []

            steps = 25
            num_points = 50

            for i in range(len(indices)-1):
                idx = indices[i]
                next_idx = indices[i+1]
                # Fitar fixando as tangents: TO DO
                #curve_bezier = fit_directly_bezier(P[idx:next_idx+1], steps, tangent_points[idx], tangent_points[next_idx])
                if next_idx == idx + 1:
                    curve_bezier = np.array([P[idx], P[idx], P[next_idx], P[next_idx]])
                else:
                    curve_bezier = Bezier.fit_directly_bezier(P[idx:next_idx+1], steps)
                curve_points, _ = Bezier.extract_points_bezier(curve_bezier, num_points)
                points_from_knots_bezier.append(curve_points)

            points_from_knots_bezier = np.concatenate(points_from_knots_bezier, axis=0)

                # Plotagem
            ax.plot(X, Y, label='Dados Originais')

            for i, (x, y) in enumerate(zip(X, Y)):
                if i < len(tangent_points):  # Ensure we have a tangent for this point
                    dx, dy = tangent_points[i]
                    ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='blue', label='Tangente' if i == 0 else "")
            
            ax.set_title(f'ε = {epsilon:.2f}')
            ax.scatter(knots[:,0], knots[:,1], label='Pontos de controle',color='orange')
            ax.plot(points_from_knots_bezier[:,0], points_from_knots_bezier[:,1], label='Pontos da curva',color='red')
            ax.legend()
            plt.axis('equal')
            #plt.show()
            print(epsilon)

        # Crie a animação
        anim = animation.FuncAnimation(fig, update, frames=range(10), interval=1000)

        # Salvar como GIF
        anim.save('RDP_epsilon.gif', writer='pillow')

        #plt.show()


    else:

        
        extracted_coefficients = extracted_coefficients_list[0]
        P, _ = Bezier.extract_points_bezier(extracted_coefficients, n=50)


        


if "__main__" == __name__:

    main()
