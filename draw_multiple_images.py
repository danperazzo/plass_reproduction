import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

from src.bezierCurveFitter import BezierCurveFitter

def write_bezier_curves(bezier_curves, filename):
    with open(filename, 'w') as f:
        for curve in bezier_curves:
            f.write(' '.join(map(str, curve)) + '\n')
    print(f"Bezier curves written to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Visualize the evolution of a Bezier curve fitting process.')
    parser.add_argument('--window_size', type=int, default=10, help='Size of the sliding window for tangent calculation.')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps for the fitting process.')
    parser.add_argument('--input_file', type=str, default='examples/g-150.txt', help='Path to the input file containing points.')
    parser.add_argument('--input_error', type=str, default='', help='Path to the input file containing error matrix.')
    parser.add_argument('--output_file', type=str, default='reconst_3.svg', help='Path to save the output animation.')
    parser.add_argument('--output_bezier', type=str, default='bezier_curves.txt', help='Path to save the Bezier curves.')
    parser.add_argument('--epsilon', type=int, default=2, help='Epsilon value for the RDP algorithm.')
    parser.add_argument('--disable_rdp',type=bool, default=False, help='Use RDP compression for the curve fitting.')
    parser.add_argument('--tolerance', type=float, default=0.005, help='Tolerance for the dynamic programming algorithm.')

    args = parser.parse_args()


    # Pegar os pontos
    points = np.loadtxt(args.input_file, delimiter=' ')  # Adjust delimiter if needed
    X = points[:, 0]
    Y = points[:, 1]

    P = np.column_stack((X, Y))


    # --- Código inicial ---
    Bezier = BezierCurveFitter(P)

    Bezier.rdp(args.epsilon, [0, len(P) - 1])

    Bezier.get_corners()

    Corners = [Bezier.knots_idx[i] for i in Bezier.corners]

    Bezier.sliding_window(args.num_steps, window_size=args.window_size)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    x_min, x_max = np.min(X) - 10, np.max(X) + 10
    y_min, y_max = np.min(Y) - 10, np.max(Y) + 10

    ax1.set_xlim(x_min - 1, x_max + 1)  
    ax1.set_ylim(y_min - 1, y_max + 1)
    ax1.set_aspect('equal')

    ax2.set_xlim(x_min - 1, x_max + 1)  
    ax2.set_ylim(y_min - 1, y_max + 1)
    ax2.set_aspect('equal')

    steps = args.num_steps
    num_points = 50

    if args.input_error == "":
        Bezier.make_error_matrix(steps)
    else:
        Bezier.error_matrix = np.load(args.input_error)

    Bezier.get_knots(args.tolerance)

    points_from_knots_bezier = []
    bezier_curves = []

    for i in range(len(Bezier.final_knots_idx) - 1):
        idx = Bezier.final_knots_idx[i]
        next_idx = Bezier.final_knots_idx[i + 1]
        if np.array_equal(Bezier.tangent_points[idx], np.array([0, 0])) and np.array_equal(Bezier.tangent_points[next_idx], np.array([0, 0])):
            curve_bezier = Bezier.fit_directly_bezier(P[idx:next_idx + 1], steps, no_fixed_endpoints=False)
        elif np.array_equal(Bezier.tangent_points[idx], np.array([0, 0])):
            curve_bezier, _ = Bezier.fit_fixed_bezier_single_gradient(P[idx:next_idx + 1], steps, Bezier.tangent_points[next_idx], False)
        elif np.array_equal(Bezier.tangent_points[next_idx], np.array([0, 0])):
            curve_bezier, _ = Bezier.fit_fixed_bezier_single_gradient(P[idx:next_idx + 1], steps, Bezier.tangent_points[idx], True)
        else:
            curve_bezier, _ = Bezier.fit_fixed_bezier(P[idx:next_idx + 1], steps, Bezier.tangent_points[idx], Bezier.tangent_points[next_idx])
        bezier_curves.append(curve_bezier)
        curve_points, _ = Bezier.extract_points_bezier(curve_bezier, num_points)
        points_from_knots_bezier.append(curve_points)

    points_from_knots_bezier = np.concatenate(points_from_knots_bezier, axis=0)

    ax1.plot(X, Y, label='Dados Originais')

    ax2.plot(X, Y, label='Dados Originais')


    ax1.set_title(f'1 - ε = {args.epsilon:.2f}')
    ax1.plot(points_from_knots_bezier[:, 0], points_from_knots_bezier[:, 1], label='Pontos da curva', color='red')

    ax1.plot(X[Bezier.knots_idx], Y[Bezier.knots_idx], 'o', label='Pontos do RDP', color='blue')

    #ax1.plot(X[Corners], Y[Corners], 'o', label='Pontos do corners', color='purple')

    #ax1.plot(Bezier.final_knots[:, 0], Bezier.final_knots[:, 1],'o', label='Pontos de controle', color='orange')

    ax2.set_title(f'2 - ε = {args.epsilon:.2f}')
    ax2.plot(points_from_knots_bezier[:, 0], points_from_knots_bezier[:, 1], label='Pontos da curva', color='red')

    #ax2.plot(X[Bezier.knots_idx], Y[Bezier.knots_idx], 'o', label='Pontos do RDP', color='blue')

    ax2.plot(X[Corners], Y[Corners], 'o', label='Pontos do corners', color='purple')

    #ax2.plot(Bezier.final_knots[:, 0], Bezier.final_knots[:, 1],'o', label='Pontos de controle', color='orange')

    plt.show()
        


if "__main__" == __name__:

    main()