import numpy as np
import math
from tqdm import tqdm

from src.utils import solve_linear_regression_fixed_points_and_gradient, solve_linear_regression_one_tangent

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
    
    def solve_linear_regression_fixed_points(self,points, matrix_t):
        initial_point = points[0:1,:]
        final_point = points[-1:,:]

        factor_initial_point = matrix_t[:,0:1] * initial_point
        factor_final_point = matrix_t[:,-1:] * final_point

        matrix_t_no_endpoints = matrix_t[:, 1:-1]

        points_translated = points - factor_final_point - factor_initial_point
        coeficients_not_fixed = self.solve_linear_regression(points_translated, matrix_t_no_endpoints)

        final_coefficients = np.vstack([initial_point, coeficients_not_fixed, final_point])

        return final_coefficients 

    def solve_linear_regression(self, P, matrix_t):
        A = matrix_t.T @ matrix_t
        b = matrix_t.T @ P

        if np.linalg.matrix_rank(A) == A.shape[0]:
            coefficients = np.linalg.solve(A, b)
        else:
            coefficients, *_ = np.linalg.lstsq(matrix_t, P, rcond=None)

        return coefficients

    
    def fit_directly_bezier(self, P, num_steps, no_fixed_endpoints=True, return_T=False):
        if P.shape[0] == 2:
            P_1 = P[0] + (P[1] - P[0]) / 3
            P_2 = P[0] + 2*(P[1] - P[0]) / 3

            if return_T:
                return np.array([P[0], P_1, P_2, P[1]]), self.initialize_T(P)
            else:
                return np.array([P[0], P_1, P_2, P[1]])

        T  = self.initialize_T(P)

        for i in range(num_steps):
            matrix_t = self.construct_matrix_T(T, 3)

            if no_fixed_endpoints:
                bezier_coefficients = self.solve_linear_regression(P, matrix_t)

            else: 
                bezier_coefficients = self.solve_linear_regression_fixed_points(P, matrix_t)
            T = self.update_T(P, T, bezier_coefficients)

        if return_T:
            return bezier_coefficients, T

        return bezier_coefficients
    

    def fit_fixed_bezier_single_gradient(self, P, num_steps, gradient, use_left=True):
        if P.shape[0] == 2:
            P_1 = P[0] + (P[1] - P[0]) / 3
            P_2 = P[0] + 2*(P[1] - P[0]) / 3

            return np.array([P[0], P_1, P_2, P[1]]), self.initialize_T(P)

        T  = self.initialize_T(P)
        grad_x = [gradient[0]]
        grad_y = [gradient[1]]

        for i in range(num_steps):
            bezier_coefficients = solve_linear_regression_one_tangent(T, P, grad_x, grad_y, use_left=use_left)
            T = self.update_T(P, T, bezier_coefficients)

        return bezier_coefficients, T

    def fit_fixed_bezier(self, P, num_steps, gradient_left, gradient_right):

        if P.shape[0] == 2:
            P_1 = P[0] + (P[1] - P[0]) / 3
            P_2 = P[0] + 2*(P[1] - P[0]) / 3

            return np.array([P[0], P_1, P_2, P[1]]), self.initialize_T(P)

        T  = self.initialize_T(P)
        grad_x = [gradient_left[0], gradient_right[0]]
        grad_y = [gradient_left[1], gradient_right[1]]


        for i in range(num_steps):
            bezier_coefficients = solve_linear_regression_fixed_points_and_gradient(T, P, grad_x, grad_y)
            T = self.update_T(P, T, bezier_coefficients)

        return bezier_coefficients, T
    

    def extract_points_bezier(self, coefficients, n=50):
        T = np.linspace(0, 1, n)
        
        T_cubic = self.construct_matrix_T(T, 3)

        points = T_cubic @ coefficients

        return points, T
    

    def extract_tangent_of_bezier(self, coefficients, T):
        d1_bezier = self.d1 @ coefficients
        
        T_quad = self.construct_matrix_T(T, 2)
        
        tangents = T_quad @ d1_bezier

        return tangents

    
    def sliding_window(self, knots, corners, num_steps, window_size=5):
        n = len(self.points)
        c = len(corners)
        fitted_curves = []
        tangent_points = []

        if c == 0:
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
        else:
            for i in tqdm(range(c), desc="Fitting Bezier curves"):
                left = knots[corners[i]]
                right = knots[corners[(i+1) % c]]

                if i != c - 1:
                    m = len(self.points[left:right + 1])
                else:
                    m = len(self.points[left:]) + len(self.points[:right + 1])

                tangent_points.append([0, 0])

                for j in range(m-2):
                    # Não queremos problema caso left > right
                    left_interval = (max(left, left + j + 1 - window_size) % n)

                    if left + j + 1 + window_size < n and left > right:
                        right_interval = left + j + 1 + window_size
                    else:
                        right_interval = (min(right, left + j + 1 + window_size - 1) % n) + 1

                    if left < right:
                        P_window = self.points[left_interval:right_interval].copy()
                        T_window = self.T[left_interval:right_interval].copy()
                    else:
                        P_window = np.concatenate([self.points[left_interval:].copy(), self.points[:right_interval].copy()], axis=0)
                        T_window = self.T[left_interval:].copy() + self.T[:right_interval].copy()

                    if np.max(T_window) != np.min(T_window):
                        T_window = (T_window - np.min(T_window)) / (np.max(T_window) - np.min(T_window))
                    else:
                        T_window = (T_window - np.min(T_window))
            
                    T_center = self.T[(left + j + 1) % n]

                    fitted_curve = self.fit_directly_bezier(P_window, num_steps)
                    fitted_curves.append(fitted_curve)
                    tangent_points.append(self.extract_tangent_of_bezier(fitted_curve, T_center)[0])
                
        tangent_points = np.array(tangent_points)/20
        return fitted_curves, tangent_points
    

    def rdp(self, points, epsilon, interval):
        """
        Ramer-Douglas-Peucker algorithm to simplify a curve.

        Parameters:
            points (numpy.ndarray): A 2D array of points (shape: [n, 2]).
            epsilon (float): The distance threshold for simplification.
            method (str): Method to choose the subdivision point. 
                        "max" for maximum error, "middle" for the middle point.

        Returns:
            list: A simplified list of indexes.
        """
        # Find the point with the maximum distance from the line connecting the first and last points
        start, end = points[interval[0]], points[interval[1]]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            line_len = 1  # Avoid division by zero

        # Compute perpendicular distances
        distances = np.abs(np.cross(points[interval[0]:interval[1] + 1] - start, line_vec) / line_len)

        # Find the index of the point with the maximum distance
        max_idx = np.argmax(distances)
        max_dist = distances[max_idx]
        
        # If the maximum distance (or middle point distance) is greater than epsilon, recursively simplify
        if max_dist > epsilon:
            # Recursively simplify the two segments
            left = self.rdp(points, epsilon, [interval[0], interval[0] + max_idx])
            right = self.rdp(points, epsilon, [interval[0] + max_idx, interval[1]])

            # Combine the results, excluding the duplicate indexes at the junction
            return left[:-1] + right
        else:
            # If no point is farther than epsilon, return the index of the endpoints
            return interval


    ######################## TO TEST ########################
    # Corners são indices de knots
    def get_corners(self, P, knots, cos=math.cos((3 * math.pi) / 4)):
        k = len(knots)

        corners = []

        for i in range(k):

            previous_idx = knots[i - 1]
            current_idx = knots[i]
            next_idx = knots[(i + 1) % k]

            # P: previus / C: current / N: next
            PC = P[previous_idx] - P[current_idx]
            NC = P[next_idx] - P[current_idx]

            mag_PC = np.linalg.norm(PC)
            mag_NC = np.linalg.norm(NC)

            cos_knot = np.dot(PC, NC) / (mag_PC * mag_NC)


            if cos_knot < cos:
                corners.append(i)

        return corners


    def bezier_dist(self, P, T, coefficients):
        matrix_t = self.construct_matrix_T(T, 3)
        bezier_points = matrix_t @ coefficients
        distances = np.linalg.norm(P - bezier_points, axis=1)
        return distances

    
    def get_knots(self, P, tangents, epsilon, steps = 5, disable_rdp=False,  tolerance=0.5):
        n = len(P)

        if not disable_rdp:
            poli_knots_idx = self.rdp(P, epsilon, [0, n - 1])
            corners = self.get_corners(P, poli_knots_idx)

        else:
            poli_knots_idx = np.arange(0, n, 1)

        m = len(poli_knots_idx)

        error_matrix = np.zeros((m, m), dtype=float)

        for i in tqdm(range(m), desc="Creating the error matrix"):
            left = poli_knots_idx[i]
            gradient_left = tangents[left]

            for j in range(i + 1, m):
                # Todas as curvas de bezier de P_i até todos os próximos pontos
                right = poli_knots_idx[j]
                gradient_right = tangents[right]

                if np.array_equal(gradient_left, np.array([0, 0])) or np.array_equal(gradient_right, np.array([0, 0])):
                    # $$$$$
                    coefficients, T_new = self.fit_directly_bezier(P[left:right+1], steps, no_fixed_endpoints=False, return_T=True)

                elif np.array_equal(gradient_left, np.array([0, 0])):
                    coefficients, T_new = self.fit_fixed_bezier_single_gradient(P[left:right+1], steps, gradient_right, use_left=False)
                elif np.array_equal(gradient_right, np.array([0, 0])):
                    coefficients, T_new = self.fit_fixed_bezier_single_gradient(P[left:right+1], steps, gradient_left, use_left=True)
                
                else:
                    coefficients, T_new = self.fit_fixed_bezier(P[left:right+1], steps, gradient_left, gradient_right)
                
                distances = self.bezier_dist(P[left:right+1], T_new, coefficients)
                error_matrix[i, j] = (distances**2).sum() if distances.size > 2 else 0.0

        dp = np.full(m, np.inf)
        prev = np.full(m, -1, dtype=int)
        dp[0] = 0

        for j in range(1, m):
            for i in range(j):
                cost = dp[i] + tolerance + error_matrix[i, j]
                if cost < dp[j]:
                    dp[j] = cost
                    prev[j] = i

        # Backtrack to find indices of kept points
        indices = []
        cur = m - 1
        while cur != 0:
            indices.append(poli_knots_idx[cur])
            cur = prev[cur]
        indices.append(0)
        indices.reverse()

        knots = [P[idx] for idx in indices]
        return knots, indices, poli_knots_idx
