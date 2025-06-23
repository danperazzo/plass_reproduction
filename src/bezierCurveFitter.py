import numpy as np
import math
from tqdm import tqdm

from src.utils import solve_linear_regression_fixed_points_and_gradient, solve_linear_regression_one_tangent, encontrar_bezier_otima

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
            print("Fitting directly Bezier with 2 points")
            P_1 = P[0] + (P[1] - P[0]) / 3
            P_2 = P[0] + 2*(P[1] - P[0]) / 3

            if return_T:
                return np.array([P[0], P_1, P_2, P[1]]), self.initialize_T(P)
            else:
                return np.array([P[0], P_1, P_2, P[1]])
        elif P.shape[0] == 3:
            print("Fitting directly Bezier with 3 points")
            # Se você quer garantir que passa por P[1] no parâmetro t calculado:
            T = self.initialize_T(P)
            t = T[1]  # parâmetro do ponto médio
            
            # Ajustar os pesos baseado no t real
            # Para uma quadrática que passa por P[1] em t:
            # Q1 = (P[1] - (1-t)²*P[0] - t²*P[2]) / (2*t*(1-t))
            
            if abs(2*t*(1-t)) > 1e-10:
                Q1 = (P[1] - (1-t)**2*P[0] - t**2*P[2]) / (2*t*(1-t))
                P_1 = (1/3)*P[0] + (2/3)*Q1
                P_2 = (2/3)*Q1 + (1/3)*P[2]
            else:
                # Fallback para sua solução original
                P_1 = P[0] + 2*(P[1] - P[0]) / 3
                P_2 = P[2] + 2*(P[1] - P[2]) / 3
            
            if return_T:
                return np.array([P[0], P_1, P_2, P[2]]), T
            else:
                return np.array([P[0], P_1, P_2, P[2]])

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
        T  = self.initialize_T(P)


        if P.shape[0] == 2:
            tangent = np.array(gradient) / np.linalg.norm(gradient)
            
            if use_left:
                # Tangente esquerda fixa
                # P1 = P[0] + alpha * tangent (alpha a determinar)
                # P2 livre, mas vamos colocá-lo alinhado com P1-P[1]
                
                # Estimar alpha baseado na distância entre pontos
                dist = np.linalg.norm(P[1] - P[0])
                alpha = dist / 3  # Estimativa inicial
                
                P1 = P[0] + alpha * tangent
                # P2 na linha entre P1 e P[1], mais próximo de P[1]
                P2 = P1 + (2/3) * (P[1] - P1)
            else:
                # Tangente direita fixa
                # P2 = P[1] - beta * tangent (beta a determinar)
                # P1 livre, mas vamos colocá-lo alinhado com P[0]-P2
                
                dist = np.linalg.norm(P[1] - P[0])
                beta = dist / 3
                
                P2 = P[1] - beta * tangent
                # P1 na linha entre P[0] e P2, mais próximo de P[0]
                P1 = P[0] + (1/3) * (P2 - P[0])
            
            return np.array([P[0], P1, P2, P[1]]), T
        
        elif P.shape[0] == 3:
            tangent = np.array(gradient) / np.linalg.norm(gradient)
            
            # Calcular parâmetro t do ponto médio
            d1 = np.linalg.norm(P[1] - P[0])
            d2 = np.linalg.norm(P[2] - P[1])
            t_mid = d1 / (d1 + d2) if (d1 + d2) > 0 else 0.5
            
            if use_left:
                # C0 = P[0], C3 = P[2]
                # C1 = P[0] + alpha * tangent (fixo em direção)
                # C2 = ? (a determinar)
                
                # Para Bézier em t_mid: B(t_mid) = P[1]
                t = t_mid
                B0 = (1-t)**3
                B1 = 3*(1-t)**2*t
                B2 = 3*(1-t)*t**2
                B3 = t**3
                
                # B0*P[0] + B1*(P[0] + alpha*tangent) + B2*C2 + B3*P[2] = P[1]
                # B1*alpha*tangent + B2*C2 = P[1] - (B0+B1)*P[0] - B3*P[2]
                
                # Temos 2 equações (x,y) e 3 incógnitas (alpha, C2_x, C2_y)
                # Adicionar restrição: C2 na direção P[2]->P[1]
                direction_C2 = P[1] - P[2]
                if np.linalg.norm(direction_C2) > 0:
                    direction_C2 = direction_C2 / np.linalg.norm(direction_C2)
                else:
                    direction_C2 = np.array([1, 0])
                
                # C2 = P[2] + beta * direction_C2
                # Substituindo:
                # B1*alpha*tangent + B2*(P[2] + beta*direction_C2) = target
                # B1*alpha*tangent + B2*beta*direction_C2 = target - B2*P[2]
                
                target = P[1] - (B0+B1)*P[0] - B3*P[2]
                rhs = target - B2*P[2]
                
                # Sistema 2x2: [tangent, direction_C2] * [B1*alpha, B2*beta]' = rhs
                A = np.column_stack([B1*tangent, B2*direction_C2])
                params = np.linalg.lstsq(A, rhs, rcond=None)[0]
                alpha, beta = params
                
                alpha = max(0.1, abs(alpha))
                beta = max(0.1, abs(beta))
                
                C0 = P[0]
                C1 = P[0] + alpha * tangent
                C2 = P[2] + beta * direction_C2
                C3 = P[2]
                
            else:
                # Tangente direita fixa
                # C2 = P[2] - beta * tangent
                # C1 = ? (a determinar)
                
                t = t_mid
                B0 = (1-t)**3
                B1 = 3*(1-t)**2*t
                B2 = 3*(1-t)*t**2
                B3 = t**3
                
                # Direção para C1: P[0]->P[1]
                direction_C1 = P[1] - P[0]
                if np.linalg.norm(direction_C1) > 0:
                    direction_C1 = direction_C1 / np.linalg.norm(direction_C1)
                else:
                    direction_C1 = np.array([1, 0])
                
                # C1 = P[0] + alpha * direction_C1
                # B1*(P[0] + alpha*direction_C1) + B2*(P[2] - beta*tangent) = P[1] - B0*P[0] - B3*P[2]
                
                target = P[1] - B0*P[0] - B3*P[2]
                rhs = target - B1*P[0] - B2*P[2]
                
                A = np.column_stack([B1*direction_C1, -B2*tangent])
                params = np.linalg.lstsq(A, rhs, rcond=None)[0]
                alpha, beta = params
                
                alpha = max(0.1, abs(alpha))
                beta = max(0.1, abs(beta))
                
                C0 = P[0]
                C1 = P[0] + alpha * direction_C1
                C2 = P[2] - beta * tangent
                C3 = P[2]
            
            return np.array([C0, C1, C2, C3]), T
            

        grad_x = [gradient[0], gradient[0]]
        grad_y = [gradient[1], gradient[1]]

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

        for i in range(num_steps):
            bezier_coefficients = encontrar_bezier_otima(P, T, gradient_left, gradient_right)
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

    
    def sliding_window(self, num_steps, window_size=5):
        n = len(self.points)
        c = len(self.corners)
        fitted_curves = []
        tangent_points = []

        if c == 0:
            for i in tqdm(range(n), desc="Fitting Bezier curves"):

                left_interval = (i - window_size) % n
                right_interval = (i + window_size) % n + 1

                center = window_size

                if left_interval < right_interval:
                    P_window = self.points[left_interval:right_interval].copy()
                else:
                    P_window = np.concatenate([self.points[left_interval:].copy(), self.points[:right_interval].copy()], axis=0)

                fitted_curve, T_window = self.fit_directly_bezier(P_window, num_steps, return_T=True)

                T_center = T_window[center]
                
                fitted_curves.append(fitted_curve)
                tangent_points.append(self.extract_tangent_of_bezier(fitted_curve, T_center)[0])
        else:
            print(f"Tem {c} corners")
            for i in tqdm(range(c), desc="Fitting Bezier curves"):
                left = self.knots_idx[self.corners[i]]
                right = self.knots_idx[self.corners[(i+1) % c]]

                if i != c - 1:
                    m = len(self.points[left:right + 1])
                else:
                    m = len(self.points[left:]) + len(self.points[:right + 1])

                # Tangente do corner
                tangent_points.append([0, 0])

                for j in range(1, m - 1):
                    # Não queremos problema caso left > right
                    if left < right:
                        left_interval = max(left, left + j - window_size)
                        right_interval = min(right, right - (m - 1 - j) + window_size) + 1
                    else:
                        left_interval = (max(left, left + j - window_size) % n)
                        right_interval = (min(right, right - (m - 1 - j) + window_size) % n) + 1

                    if left_interval != left:
                        center = window_size
                    else:
                        center = j

                    if left_interval < right_interval:
                        P_window = self.points[left_interval:right_interval].copy()
                    else:
                        P_window = np.concatenate([self.points[left_interval:].copy(), self.points[:right_interval].copy()], axis=0)
            
                    fitted_curve, T_window = self.fit_directly_bezier(P_window, num_steps, return_T=True)

                    T_center = T_window[center]

                    fitted_curves.append(fitted_curve)
                    tangent_points.append(self.extract_tangent_of_bezier(fitted_curve, T_center)[0])
            tangent_points = tangent_points[-self.knots_idx[self.corners[0]]:] + tangent_points[:-self.knots_idx[self.corners[0]]]
                
        tangent_points = np.array(tangent_points)/20
        self.fitted_curves = fitted_curves
        self.tangent_points = tangent_points
    

    def do_rdp(self, epsilon, interval):
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
        start, end = self.points[interval[0]], self.points[interval[1]]

        if np.array_equal(start, end):
            distances = np.linalg.norm(self.points[interval[0]:interval[1] + 1] - start, axis=1)
        else:
            line_vec = end - start
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                line_len = 1  # Avoid division by zero

            # Compute perpendicular distances
            distances = np.abs(np.cross(self.points[interval[0]:interval[1] + 1] - start, line_vec) / line_len)

        # Find the index of the point with the maximum distance
        max_idx = np.argmax(distances)
        max_dist = distances[max_idx]
        
        # If the maximum distance (or middle point distance) is greater than epsilon, recursively simplify
        if max_dist > epsilon:
            # Recursively simplify the two segments
            left = self.do_rdp(epsilon, [interval[0], interval[0] + max_idx])
            right = self.do_rdp(epsilon, [interval[0] + max_idx, interval[1]])

            # Combine the results, excluding the duplicate indexes at the junction
            return left[:-1] + right
        else:
            # If no point is farther than epsilon, return the index of the endpoints
            return interval
        

    def rdp(self, epsilon, interval):
        self.knots_idx = self.do_rdp(epsilon, interval)


    # Corners são indices de knots
    def get_corners(self, cos=math.cos((3 * math.pi) / 4)):
        k = len(self.knots_idx)

        corners = []

        for i in range(k):

            previous_idx = self.knots_idx[i - 1]
            current_idx = self.knots_idx[i]
            next_idx = self.knots_idx[(i + 1) % k]

            if np.array_equal(self.points[previous_idx], self.points[current_idx]):
                previous_idx = self.knots_idx[(i - 2) % k]
            elif np.array_equal(self.points[next_idx], self.points[current_idx]):
                next_idx = self.knots_idx[(i + 2) % k]


            # P: previus / C: current / N: next
            PC = self.points[previous_idx] - self.points[current_idx]
            NC = self.points[next_idx] - self.points[current_idx]

            mag_PC = np.linalg.norm(PC)
            mag_NC = np.linalg.norm(NC)

            cos_knot = np.dot(PC, NC) / (mag_PC * mag_NC)


            if cos_knot > cos:
                corners.append(i)

        self.corners = corners


    def bezier_dist(self, P, T, coefficients):
        matrix_t = self.construct_matrix_T(T, 3)
        bezier_points = matrix_t @ coefficients
        distances = np.linalg.norm(P - bezier_points, axis=1)
        return distances
    

    def make_error_matrix(self, steps = 5):
        m = len(self.knots_idx)

        error_matrix = np.zeros((m, m), dtype=float)

        for i in tqdm(range(m), desc="Creating the error matrix"):
            left = self.knots_idx[i]
            gradient_left = self.tangent_points[left]

            for j in range(i + 1, m):
                # Todas as curvas de bezier de P_i até todos os próximos pontos
                right = self.knots_idx[j]
                gradient_right = self.tangent_points[right]

                if np.array_equal(gradient_left, np.array([0, 0])) and np.array_equal(gradient_right, np.array([0, 0])):
                    coefficients, T_new = self.fit_directly_bezier(self.points[left:right+1], steps, no_fixed_endpoints=False, return_T=True)
                elif np.array_equal(gradient_left, np.array([0, 0])):
                    coefficients, T_new = self.fit_fixed_bezier_single_gradient(self.points[left:right+1], steps, gradient_right, use_left=False)
                elif np.array_equal(gradient_right, np.array([0, 0])):
                    coefficients, T_new = self.fit_fixed_bezier_single_gradient(self.points[left:right+1], steps, gradient_left, use_left=True)
                else:
                    coefficients, T_new = self.fit_fixed_bezier(self.points[left:right+1], steps, gradient_left, gradient_right)

                
                distances = self.bezier_dist(self.points[left:right+1], T_new, coefficients)
                error_matrix[i, j] = (distances**2).sum() if distances.size > 2 else 0.0
                
        
        self.error_matrix = error_matrix


    def get_knots(self, tolerance = 0.5):
        m = len(self.knots_idx)

        dp = np.full(m, np.inf)
        prev = np.full(m, -1, dtype=int)
        dp[0] = 0

        for j in range(1, m):
            for i in range(j):
                cost = dp[i] + tolerance + self.error_matrix[i, j]
                if cost < dp[j]:
                    dp[j] = cost
                    prev[j] = i

        # Backtrack to find indices of kept points
        indices = []
        cur = m - 1
        while cur != 0:
            indices.append(self.knots_idx[cur])
            cur = prev[cur]
        indices.append(0)
        indices.reverse()

        self.final_knots = np.array([self.points[idx] for idx in indices])
        self.final_knots_idx = indices
