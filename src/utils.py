import numpy as np


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