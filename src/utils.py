import numpy as np


import numpy as np

def fit_bezier(points, parameters_t, initial_tangent, final_tangent):
    """
    Fits a cubic Bézier curve with fixed endpoints and fixed tangent directions.
    """

    # Convert inputs to contiguous float64 arrays
    points = np.asarray(points, dtype=np.float64)
    t = np.asarray(parameters_t, dtype=np.float64)
    v1 = np.asarray(initial_tangent, dtype=np.float64)
    v2 = np.asarray(final_tangent, dtype=np.float64)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    P0 = points[0]
    P3 = points[-1]
    n = points.shape[0]

    # Cubic Bernstein basis
    mt = 1.0 - t
    B0 = mt * mt * mt
    B1 = 3.0 * mt * mt * t
    B2 = 3.0 * mt * t * t
    B3 = t * t * t

    # Known term of the system
    S1 = B0 + B1
    S2 = B2 + B3
    known = S1[:, None] * P0 + S2[:, None] * P3

    # Assemble Ax/Ay and Bx/By coefficients
    A = np.empty((2 * n, 2), dtype=np.float64)
    A[:n, 0] = B1 * v1[0]
    A[n:, 0] = B1 * v1[1]
    A[:n, 1] = -B2 * v2[0]
    A[n:, 1] = -B2 * v2[1]

    # Right-hand side vector
    b = np.empty(2 * n, dtype=np.float64)
    b[:n] = points[:, 0] - known[:, 0]
    b[n:] = points[:, 1] - known[:, 1]

    # Least-squares solution
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    alpha, beta = x

    # Control points
    ctrl = np.empty((4, points.shape[1]), dtype=np.float64)
    ctrl[0] = P0
    ctrl[1] = P0 + alpha * v1
    ctrl[2] = P3 - beta * v2
    ctrl[3] = P3

    return ctrl


def fit_bezier_linear_one_tangent(t_data, X_data, Y_data,
                                   x_0, x_m, y_0, y_m,
                                   vx, vy, use_left=True):
    t = t_data
    t1 = 1 - t
    b0 = t1**3
    b1 = 3 * t * t1**2
    b2 = 3 * t**2 * t1
    b3 = t**3

    P0 = np.array([x_0, y_0])
    P3 = np.array([x_m, y_m])

    if use_left:
        # B1 = P0 + alpha * v / 3, B2 is free
        A = np.zeros((2 * len(t), 3))  # columns: alpha, B2_x, B2_y
        rhs = np.zeros(2 * len(t))

        B1_term_x = (vx[0] / 3.0) * b1
        B1_term_y = (vy[0] / 3.0) * b1

        known_x = b0 * x_0 + b3 * x_m + b1 * x_0
        known_y = b0 * y_0 + b3 * y_m + b1 * y_0

        A[0::2, 0] = B1_term_x           # alpha
        A[0::2, 1] = b2                  # B2_x
        rhs[0::2] = X_data - known_x     # target x - known

        A[1::2, 0] = B1_term_y
        A[1::2, 2] = b2                  # B2_y
        rhs[1::2] = Y_data - known_y

    else:
        # B2 = P3 - alpha * v / 3, B1 is free
        A = np.zeros((2 * len(t), 3))  # columns: alpha, B1_x, B1_y
        rhs = np.zeros(2 * len(t))

        B2_term_x = -(vx[0] / 3.0) * b2
        B2_term_y = -(vy[0] / 3.0) * b2

        known_x = b0 * x_0 + b3 * x_m + b2 * x_m
        known_y = b0 * y_0 + b3 * y_m + b2 * y_m

        A[0::2, 0] = B2_term_x           # alpha
        A[0::2, 1] = b1                  # B1_x
        rhs[0::2] = X_data - known_x

        A[1::2, 0] = B2_term_y
        A[1::2, 2] = b1                  # B1_y
        rhs[1::2] = Y_data - known_y

    # Solve linear least squares: A @ [alpha, free_x, free_y] = rhs
    sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    return sol  # [alpha, x_free, y_free]


def solve_linear_regression_one_tangent(t, points, gradient_x, gradient_y, use_left=True):
    """
    Fits a cubic Bézier curve given sampled points, fixed endpoints, and only one tangent direction (left or right).
    """
    initial_point = points[0, :]
    final_point = points[-1, :]

    x_0, y_0 = initial_point
    x_m, y_m = final_point

    res = fit_bezier_linear_one_tangent(
        t_data=t,
        X_data=points[:, 0],
        Y_data=points[:, 1],
        x_0=x_0, x_m=x_m,
        y_0=y_0, y_m=y_m,
        vx=gradient_x, vy=gradient_y,
        use_left=use_left
    )

    if use_left:
        alpha, B2_x, B2_y = res
        B1 = initial_point + (alpha / 3) * np.array([gradient_x[0], gradient_y[0]])
        B2 = np.array([B2_x, B2_y])
    else:
        alpha, B1_x, B1_y = res
        B2 = final_point - (alpha / 3) * np.array([gradient_x[0], gradient_y[0]])
        B1 = np.array([B1_x, B1_y])

    final_coefficients = np.vstack([initial_point, B1, B2, final_point])
    return final_coefficients




def solve_linear_regression_fixed_points_and_gradient(t, points, gradient_x, gradient_y):
    initial_point = points[0,:]
    final_point = points[-1,:]


    res = fit_bezier_tangents_simultaneous(t, points[:, 0], points[:, 1], initial_point[0], final_point[0],
                                               initial_point[1], final_point[1], gradient_x, gradient_y)
    
    B1 = initial_point + (res[0] * np.array([gradient_x[0], gradient_y[0]])) / 3
    B2 = final_point - (res[1] * np.array([gradient_x[1], gradient_y[1]])) / 3

    final_coefficients = np.vstack([initial_point, B1, B2, final_point])

    return final_coefficients

def fit_bezier_tangents_simultaneous(t_data, X_data, Y_data, x_0, x_m, y_0, y_m, vx, vy):

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

    alpha_l, alpha_r = params

    if alpha_l < 0:
        alpha_l = -alpha_l
    if alpha_r < 0:
        alpha_r = -alpha_r

    return alpha_l, alpha_r # alpha_l_fit, alpha_r_fit
