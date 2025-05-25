import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

from src.bezierCurveFitter import BezierCurveFitter

def main():
    parser = argparse.ArgumentParser(description='Visualize the evolution of a Bezier curve fitting process.')
    parser.add_argument('--window_size', type=int, default=10, help='Size of the sliding window for tangent calculation.')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps for the fitting process.')
    parser.add_argument('--input_file', type=str, default='examples/3-trace.txt', help='Path to the input file containing points.')
    parser.add_argument('--output_file', type=str, default='reconst_3.svg', help='Path to save the output animation.')
    parser.add_argument('--epsilon', type=int, default=2, help='Epsilon value for the RDP algorithm.')
    parser.add_argument('--disable_rdp',type=bool, default=False, help='Use RDP compression for the curve fitting.')
    parser.add_argument('--tolerance', type=float, default=0.5, help='Tolerance for the dynamic programming algorithm.')

    args = parser.parse_args()


    # Pegar os pontos
    points = np.loadtxt(args.input_file, delimiter=' ')  # Adjust delimiter if needed
    X = points[:, 0]
    Y = points[:, 1]

    P = np.column_stack((X, Y))


    # --- Código inicial ---
    Bezier = BezierCurveFitter(P)


    _, tangent_points = Bezier.sliding_window(args.num_steps, window_size=args.window_size)
    
    fig, ax = plt.subplots()

    x_min, x_max = np.min(X) - 10, np.max(X) + 10
    y_min, y_max = np.min(Y) - 10, np.max(Y) + 10
    ax.set_xlim(x_min - 1, x_max + 1)  
    ax.set_ylim(y_min - 1, y_max + 1)  
    ax.set_aspect('equal')

    steps = args.num_steps
    num_points = 50

    knots, indices, poli_knots_idx = Bezier.get_knots(P, tangent_points, args.epsilon, steps, args.disable_rdp, args.tolerance)
    knots = np.array(knots)

    points_from_knots_bezier = []

    for i in range(len(indices) - 1):
        idx = indices[i]
        next_idx = indices[i + 1]
        curve_bezier, _ = Bezier.fit_fixed_bezier(P[idx:next_idx + 1], steps, tangent_points[idx], tangent_points[next_idx])
        curve_points, _ = Bezier.extract_points_bezier(curve_bezier, num_points)
        points_from_knots_bezier.append(curve_points)

    points_from_knots_bezier = np.concatenate(points_from_knots_bezier, axis=0)

    ax.plot(X, Y, label='Dados Originais')

    ax.plot(X[poli_knots_idx], Y[poli_knots_idx], 'o', label='Pontos do RDP', color='blue')

    ax.set_title(f'ε = {args.epsilon:.2f}')
    ax.plot(points_from_knots_bezier[:, 0], points_from_knots_bezier[:, 1], label='Pontos da curva', color='red')
    ax.plot(knots[:, 0], knots[:, 1],'o', label='Pontos de controle', color='orange')
    plt.show()
    plt.savefig(args.output_file, format='svg', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {args.output_file}")
        


if "__main__" == __name__:

    main()
