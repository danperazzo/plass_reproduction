import numpy as np
import math
from tqdm import tqdm

from src.utils import (
    solve_linear_regression_fixed_points_and_gradient,
    solve_linear_regression_one_tangent,
    fit_bezier
)

class BezierCurveFitter:
    """Handles the fitting of Bezier curves to point sets."""

    def __init__(self, P, max_iterations=21):
        # Maximum number of Newton steps allowed for reparametrization
        self.MAX_ITERATIONS = max_iterations

        # Derivative matrices for cubic Bézier curves
        # These allow evaluating dB/dt and d²B/dt² linearly
        self.d1 = np.array([
            [-3, 3, 0, 0],
            [0, -3, 3, 0],
            [0, 0, -3, 3]
        ])
        self.d2 = np.array([
            [6, -12, 6, 0],
            [0, 6, -12, 6]
        ])

        # Original ordened list of points
        self.points = P
        self.T, self.dist = self.initialize_T(P)


    def initialize_T(self, P):
        """
        Creates an initial chord-length parameterization.

        This gives a reasonable starting point for Newton updates.

        Parameters:
            P (ndarray): Ordered sample points of the curve.

        Returns:
            T (ndarray): Parameter values normalized to [0, 1].
            d (float): Total chord-length distance from first to last point.
        """
        T = [0.0]
        s = 0.0
        n = len(P)

        for i in range(1, n):
            s += np.linalg.norm(P[i] - P[i-1])
            T.append(s)

        # d is the direct distance from last point to first point
        if n > 1:
            d = np.linalg.norm(P[-1] - P[0])
            T = np.array(T) / s
        else:
            d = 0.0

        return T, d
    

    def get_segment_points(self, left, right):
        if left <= right:
            return self.points[left:right+1]
        return np.concatenate([self.points[left:], self.points[:right+1]], axis=0)

    def get_segment_T(self, left, right):
        if left <= right:
            T_global = self.T[left:right+1]
        else:
            T_global = np.concatenate([
                self.T[left:], 
                self.T[:right+1] + self.T[-1] + 1  # ou self.dist + 1
            ])
        return (T_global - T_global[0]) / (T_global[-1] - T_global[0])
        

    def construct_matrix_T(self, t, n):
        """
        Builds the Bernstein basis matrix of degree n.
        """
        return np.vstack([
            math.comb(n, k) * ((1 - t)**(n - k)) * (t**k)
            for k in range(n + 1)
        ]).T


    def update_T(self, P, T, coefficients):
        """
        Refines T via Newton iterations.

        Notes:
            - Stops early if updates become too small.
            - Helps maintain a roughly uniform parametrization.
        """
        error_old = 1
        error = 0
        iterations = 0

        while abs(error - error_old) > 1e-7 and iterations < self.MAX_ITERATIONS:
            T_old = T
            T = self.compute_step_newton(P, T, coefficients)

            error_old = error
            error = np.sum((T - T_old)**2)
            iterations += 1

        return T


    def compute_step_newton(self, P, T, coefficients):
        """
        Performs a single Newton step for reparametrization.
        """

        # Control points of the derivative curves B'(t) and B''(t)
        d1_bezier = self.d1 @ coefficients
        d2_bezier = self.d2 @ coefficients

        # Basis matrices for evaluating B(t), B'(t), B''(t)
        T_cubic = self.construct_matrix_T(T, 3)
        T_quad  = self.construct_matrix_T(T, 2)
        T_linear = self.construct_matrix_T(T, 1)

        # Evaluate Bézier curve and its derivatives
        P1 = (coefficients.T @ T_cubic.T).T
        P2 = (d1_bezier.T @ T_quad.T).T
        P3 = (d2_bezier.T @ T_linear.T).T

        # Numerator and denominator of Newton-Raphson update
        f = np.sum((P1 - P) * P2)
        d1_f = np.sum(P2**2 + (P1 - P) * P3)

        # Robustness against flat derivatives
        if abs(d1_f) < 1e-7:
            d1_f = 1e-7

        # Newton-Raphson update
        T -= f / d1_f

        # Normalize T to keep it within a valid range
        Tmin, Tmax = np.min(T), np.max(T)
        if Tmax != Tmin:
            T = (T - Tmin) / (Tmax - Tmin)
        else:
            T = T - Tmin

        return T


    ###############################################################
    # Basic linear regression solvers
    ###############################################################
    def solve_linear_regression_fixed_points(self, points, matrix_t):
        initial_point = points[0:1,:]
        final_point = points[-1:,:]

        factor_initial_point = matrix_t[:,0:1] * initial_point
        factor_final_point = matrix_t[:,-1:] * final_point

        matrix_t_no_endpoints = matrix_t[:, 1:-1]

        points_translated = points - factor_final_point - factor_initial_point
        coef_not_fixed = self.solve_linear_regression(points_translated, matrix_t_no_endpoints)

        return np.vstack([initial_point, coef_not_fixed, final_point])


    def solve_linear_regression(self, P, matrix_t):
        A = matrix_t.T @ matrix_t
        b = matrix_t.T @ P

        if np.linalg.matrix_rank(A) == A.shape[0]:
            return np.linalg.solve(A, b)
        else:
            return np.linalg.lstsq(matrix_t, P, rcond=None)[0]

    
    def fit_directly_bezier(self, interval, num_steps, no_fixed_endpoints=True, return_T=False):
        """
        Fits a cubic Bézier curve to the given points without enforcing endpoint
        tangent constraints.

        Parameters:
            interval (list or tuple): Interval defining the domain on which the curve is evaluated.
            num_steps (int): Number of LS + reparametrization iterations.
            no_fixed_endpoints (bool): If False, forces interpolation of endpoints.
            return_T (bool): If True, returns both coefficients and T.

        Returns:
            ndarray or (ndarray, ndarray): Bézier control points (and T if requested).
        """

        left, right = interval
        P = self.get_segment_points(left, right)

        # Special case: exactly 2 points → simple linear Bézier
        if P.shape[0] == 2:
            P1 = P[0] + (P[1] - P[0]) / 3
            P2 = P[0] + 2 * (P[1] - P[0]) / 3
            curve = np.array([P[0], P1, P2, P[1]])

            T = self.get_segment_T(left, right)

            return (curve, T) if return_T else curve

        # Special case: 3 points → use quadratic midpoint relationship
        elif P.shape[0] == 3:
            T = self.get_segment_T(left, right)
            t = T[1]

            if abs(2 * t * (1 - t)) > 1e-10:
                # Compute Q1 (quadratic intermediate point)
                Q1 = (P[1] - (1 - t)**2 * P[0] - t**2 * P[2]) / (2 * t * (1 - t))
                P1 = (1/3) * P[0] + (2/3) * Q1
                P2 = (2/3) * Q1 + (1/3) * P[2]
            else:
                # Degenerate fallback
                P1 = P[0] + 2 * (P[1] - P[0]) / 3
                P2 = P[2] + 2 * (P[1] - P[2]) / 3

            curve = np.array([P[0], P1, P2, P[2]])
            return (curve, T) if return_T else curve

        # General case: alternating LS + T refinement
        T = self.get_segment_T(left, right)

        for _ in range(num_steps):
            matrix_t = self.construct_matrix_T(T, 3)

            if no_fixed_endpoints:
                bezier_coeffs = self.solve_linear_regression(P, matrix_t)
            else:
                bezier_coeffs = self.solve_linear_regression_fixed_points(P, matrix_t)

            T = self.update_T(P, T, bezier_coeffs)

        return (bezier_coeffs, T) if return_T else bezier_coeffs


    def fit_fixed_bezier_single_gradient(self, interval, num_steps, gradient, use_left=True):
        """
        Fits a cubic Bézier curve enforcing a tangent direction at only one endpoint.

        Parameters:
            interval (list or tuple): Interval defining the domain on which the curve is evaluated.
            num_steps (int): Iteration count for LS + update_T.
            gradient (array-like): Tangent direction to enforce.
            use_left (bool): If True, fix tangent at left endpoint.

        Returns:
            (coeffs, T): Fitted Bézier control points and updated parameterization.
        """
        left, right = interval
        P = self.get_segment_points(left, right)

        T = self.get_segment_T(left, right)

        # Handling very small sets separately keeps the logic clean
        if P.shape[0] == 2:
            tangent = np.array(gradient) / np.linalg.norm(gradient)

            if use_left:
                d = np.linalg.norm(P[1] - P[0])
                alpha = d / 3
                P1 = P[0] + alpha * tangent
                P2 = P1 + (2/3) * (P[1] - P1)
            else:
                d = np.linalg.norm(P[1] - P[0])
                beta = d / 3
                P2 = P[1] - beta * tangent
                P1 = P[0] + (1/3) * (P2 - P[0])

            return np.array([P[0], P1, P2, P[1]]), T

        elif P.shape[0] == 3:
            # Enforce interpolation at the middle and tangent at one endpoint.
            tangent = np.array(gradient) / np.linalg.norm(gradient)

            # Compute fractional parameter of the middle point
            d1 = np.linalg.norm(P[1] - P[0])
            d2 = np.linalg.norm(P[2] - P[1])
            t = d1 / (d1 + d2) if (d1 + d2) > 0 else 0.5

            # Bernstein basis for cubic at t
            B0 = (1 - t)**3
            B1 = 3 * (1 - t)**2 * t
            B2 = 3 * (1 - t) * t**2
            B3 = t**3

            if use_left:
                # Direction for C2 when left tangent is fixed
                direction_C2 = P[1] - P[2]
                if np.linalg.norm(direction_C2) == 0:
                    direction_C2 = np.array([1, 0])
                else:
                    direction_C2 /= np.linalg.norm(direction_C2)

                target = P[1] - (B0 + B1) * P[0] - B3 * P[2]
                rhs = target - B2 * P[2]

                # Solve for alpha, beta s.t. C1 = P0 + α*tangent, C2 = P2 + β*direction_C2
                A = np.column_stack([B1 * tangent, B2 * direction_C2])
                alpha, beta = np.linalg.lstsq(A, rhs, rcond=None)[0]

                alpha = max(0.1, abs(alpha))
                beta  = max(0.1, abs(beta))

                C0 = P[0]
                C1 = P[0] + alpha * tangent
                C2 = P[2] + beta * direction_C2
                C3 = P[2]

            else:
                # Right tangent fixed
                direction_C1 = P[1] - P[0]
                if np.linalg.norm(direction_C1) == 0:
                    direction_C1 = np.array([1, 0])
                else:
                    direction_C1 /= np.linalg.norm(direction_C1)

                target = P[1] - B0 * P[0] - B3 * P[2]
                rhs = target - B1 * P[0] - B2 * P[2]

                A = np.column_stack([B1 * direction_C1, -B2 * tangent])
                alpha, beta = np.linalg.lstsq(A, rhs, rcond=None)[0]

                alpha = max(0.1, abs(alpha))
                beta  = max(0.1, abs(beta))

                C0 = P[0]
                C1 = P[0] + alpha * direction_C1
                C2 = P[2] - beta * tangent
                C3 = P[2]

            return np.array([C0, C1, C2, C3]), T

        grad_x = [gradient[0], gradient[0]]
        grad_y = [gradient[1], gradient[1]]

        for _ in range(num_steps):
            coeffs = solve_linear_regression_one_tangent(T, P, grad_x, grad_y, use_left=use_left)
            T = self.update_T(P, T, coeffs)

        return coeffs, T


    def fit_fixed_bezier(self, interval, num_steps, gradient_left, gradient_right):
        """
        Fits a Bézier enforcing tangents at both endpoints.
        """
        left, right = interval
        P = self.get_segment_points(left, right)

        if P.shape[0] == 2:
            P1 = P[0] + (P[1] - P[0]) / 3
            P2 = P[0] + 2 * (P[1] - P[0]) / 3
            T = self.get_segment_T(left, right)
            return np.array([P[0], P1, P2, P[1]]), T

        T = self.get_segment_T(left, right)

        for _ in range(num_steps):
            coeffs = fit_bezier(P, T, gradient_left, gradient_right)
            T = self.update_T(P, T, coeffs)

        return coeffs, T


    def extract_points_bezier(self, coefficients, n=50):
        """
        Samples n points along the Bézier curve.

        Returns:
            points (ndarray), T (ndarray)
        """
        T = np.linspace(0, 1, n)
        T_cubic = self.construct_matrix_T(T, 3)
        points = T_cubic @ coefficients
        return points, T


    def extract_tangent_of_bezier(self, coefficients, T):
        """
        Computes tangents B'(T) at given parameter(s).
        """
        d1_bezier = self.d1 @ coefficients
        T_quad = self.construct_matrix_T(T, 2)
        return T_quad @ d1_bezier


    def sliding_window(self, num_steps, window_size=5):
        """
        Estimates local tangents along the curve by fitting Bézier curves
        to sliding windows of sample points.

        Parameters:
            num_steps (int): Iterations for each local Bézier fit.
            window_size (int): Half-size of the sliding window.

        Notes:
            - For smooth regions (no corners), windows wrap around the curve.
            - For segments between corners, windows are restricted to that interval.
            - The center of the window determines the tangent evaluation point.

        Sets:
            self.fitted_curves  : list of local Bézier control points
            self.tangent_points : tangent vectors at each original sample point
        """
        n = len(self.points)
        c = len(self.corners)
        fitted_curves = []
        tangent_points = []

        if c == 0:
            # Smooth case: treat as closed loop
            for i in tqdm(range(n), desc="Fitting Bezier curves"):

                left_i  = (i - window_size) % n
                right_i = (i + window_size) % n + 1
                center  = window_size

                curve, T_local = self.fit_directly_bezier([left_i, right_i], num_steps, return_T=True)
                T_center = T_local[center]

                fitted_curves.append(curve)
                tangent_points.append(
                    self.extract_tangent_of_bezier(curve, T_center)[0]
                )

        else:
            # Corner-aware mode: treat each segment independently
            for i in tqdm(range(c), desc="Fitting Bezier curves"):
                left  = self.knots_idx[self.corners[i]]
                right = self.knots_idx[self.corners[(i + 1) % c]]

                if i != c - 1:
                    m = len(self.points[left:right + 1])
                else:
                    # Wrap last corner to first
                    m = len(self.points[left:]) + len(self.points[:right + 1])

                tangent_points.append([0, 0])  # Explicit tangent placeholder at corner

                for j in range(1, m - 1):

                    if left < right:
                        li = max(left, left + j - window_size)
                        ri = min(right, right - (m - 1 - j) + window_size) + 1
                    else:
                        # Wrap-around segment
                        li = (max(left, left + j - window_size) % n)
                        ri = (min(right, right - (m - 1 - j) + window_size) % n) + 1

                    center = window_size if li != left else j

                    curve, T_local = self.fit_directly_bezier([li, ri], num_steps, return_T=True)
                    T_center = T_local[center]

                    fitted_curves.append(curve)
                    tangent_points.append(
                        self.extract_tangent_of_bezier(curve, T_center)[0]
                    )

            tangent_points = (
                tangent_points[-self.knots_idx[self.corners[0]]:] +
                tangent_points[:-self.knots_idx[self.corners[0]]]
            )

        self.fitted_curves = fitted_curves
        self.tangent_points = np.array(tangent_points) / 20


    def do_rdp(self, epsilon, interval):
        """
        Core Ramer–Douglas–Peucker simplification step.

        Parameters:
            epsilon (float): Maximum allowed deviation from the line segment.
            interval (list): [start_index, end_index] in the point sequence.

        Returns:
            list: Indices of points to keep in this interval.
        """
        start, end = self.points[interval[0]], self.points[interval[1]]

        # If endpoints coincide, simply measure distance to single reference
        if np.array_equal(start, end):
            distances = np.linalg.norm(
                self.points[interval[0]:interval[1] + 1] - start,
                axis=1
            )
        else:
            line = end - start
            L = np.linalg.norm(line)
            if L == 0:
                L = 1

            # Perpendicular distance to segment
            distances = np.abs(
                np.cross(self.points[interval[0]:interval[1] + 1] - start, line) / L
            )

        k = np.argmax(distances)
        max_dist = distances[k]

        if max_dist > epsilon:
            left = self.do_rdp(epsilon, [interval[0], interval[0] + k])
            right = self.do_rdp(epsilon, [interval[0] + k, interval[1]])
            return left[:-1] + right
        else:
            return interval


    def rdp(self, epsilon, interval):
        """Wrapper for do_rdp. Stores final knot indices."""
        self.knots_idx = self.do_rdp(epsilon, interval)


    def get_corners(self, cos=math.cos((3 * math.pi) / 4)):
        """
        Detects sharp corners by comparing angles at each knot.

        Parameters:
            cos (float): Cosine threshold for angle sharpness.

        Sets:
            self.corners (list): Indices of corner knots.
        """
        k = len(self.knots_idx)
        corners = []

        for i in range(k):
            prev_idx = self.knots_idx[i - 1]
            cur_idx  = self.knots_idx[i]
            next_idx = self.knots_idx[(i + 1) % k]

            # Handle degenerate repeated points
            if np.array_equal(self.points[prev_idx], self.points[cur_idx]):
                prev_idx = self.knots_idx[(i - 2) % k]
            elif np.array_equal(self.points[next_idx], self.points[cur_idx]):
                next_idx = self.knots_idx[(i + 2) % k]

            PC = self.points[prev_idx] - self.points[cur_idx]
            NC = self.points[next_idx] - self.points[cur_idx]

            magPC = np.linalg.norm(PC)
            magNC = np.linalg.norm(NC)

            angle_cos = np.dot(PC, NC) / (magPC * magNC)

            if angle_cos > cos:
                corners.append(i)

        self.corners = corners


    def bezier_dist(self, P, T, coefficients):
        matrix_t = self.construct_matrix_T(T, 3)
        bezier_points = matrix_t @ coefficients
        distances = np.linalg.norm(P - bezier_points, axis=1)
        return distances


    def make_error_matrix(self, steps=5):
        """
        Builds the matrix of fitting errors for all pairs of knot indices.

        Each entry (i, j) is the squared fitting error of the Bézier curve
        approximating the segment from knot i to knot j.

        Parameters:
            steps (int): Iteration count for each Bézier fit.

        Sets:
            self.error_matrix (ndarray)
        """
        m = len(self.knots_idx)
        E = np.zeros((m, m), dtype=float)

        for i in tqdm(range(m), desc="Creating the error matrix"):
            left_idx = self.knots_idx[i]
            grad_left = self.tangent_points[left_idx]

            for j in range(i + 1, m):
                right_idx = self.knots_idx[j]
                grad_right = self.tangent_points[right_idx]

                pts = self.points[left_idx:right_idx + 1]

                # Choose fitting strategy depending on tangent availability
                if np.array_equal(grad_left, [0, 0]) and np.array_equal(grad_right, [0, 0]):
                    coeffs, T_new = self.fit_directly_bezier(
                        [left_idx, right_idx], steps, no_fixed_endpoints=False, return_T=True
                    )
                elif np.array_equal(grad_left, [0, 0]):
                    coeffs, T_new = self.fit_fixed_bezier_single_gradient(
                        [left_idx, right_idx], steps, grad_right, use_left=False
                    )
                elif np.array_equal(grad_right, [0, 0]):
                    coeffs, T_new = self.fit_fixed_bezier_single_gradient(
                        [left_idx, right_idx], steps, grad_left, use_left=True
                    )
                else:
                    coeffs, T_new = self.fit_fixed_bezier(
                        [left_idx, right_idx], steps, grad_left, grad_right
                    )

                d = self.bezier_dist(pts, T_new, coeffs)
                E[i, j] = (d**2).sum() if d.size > 2 else 0.0

        self.error_matrix = E


    def get_knots(self, tolerance=0.5):
        """
        Chooses the optimal subset of knots using dynamic programming.

        Parameters:
            tolerance (float): Penalty per segment to avoid over-segmentation.

        Sets:
            self.final_knots      : Coordinates of selected knots
            self.final_knots_idx  : Indices of selected knots
        """
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

        # Backtracking
        out = []
        cur = m - 1
        while cur != 0:
            out.append(self.knots_idx[cur])
            cur = prev[cur]
        out.append(0)
        out.reverse()

        self.final_knots = np.array([self.points[idx] for idx in out])
        self.final_knots_idx = out