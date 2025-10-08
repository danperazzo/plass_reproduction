import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import argparse

from src.bezierCurveFitter import BezierCurveFitter


def main():
    parser = argparse.ArgumentParser(description='Visualize the evolution of a Bezier curve fitting process.')
    parser.add_argument('--window_size', type=int, default=5, help='Size of the sliding window for tangent calculation. (Default 5)')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of steps for the fitting process. (Default 10)')
    parser.add_argument('--input_file', type=str, default='examples/C.txt', help='Path to the input file containing points.')
    parser.add_argument('--input_error', type=str, default='', help='Path to the input file containing error matrix.')
    parser.add_argument('--epsilon', type=float, default=1, help='Epsilon value for the RDP algorithm (default 1)')
    parser.add_argument('--tau_tolerance', type=float, default=0.005, help='Tau tolerance for the dynamic programming algorithm. (default 0.005)')

    args = parser.parse_args()


    # Get the points
    points = np.loadtxt(args.input_file, delimiter=' ')  # Adjust delimiter if needed
    X = points[:, 0]
    Y = points[:, 1]

    P = np.column_stack((X, Y))


    # --- Initial Code ---
    Bezier = BezierCurveFitter(P)


    # Sliders
    ax_slider_tolerance = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider_tolerance = Slider(ax_slider_tolerance, 'τ', 0.0, 5.0, valinit=args.tau_tolerance, valstep=0.01)

    
    fig, ax = plt.subplots()

    x_min, x_max = np.min(X) - 10, np.max(X) + 10
    y_min, y_max = np.min(Y) - 10, np.max(Y) + 10
    ax.set_xlim(x_min - 1, x_max + 1)  
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_aspect('equal')


    steps = args.num_steps
    num_points = 150
    
    Bezier.rdp(args.epsilon, [0, len(P) - 1])

    Bezier.get_corners()

    Bezier.sliding_window(steps, args.window_size)

    if args.input_error == "":
        Bezier.make_error_matrix(steps)
    else:
        Bezier.error_matrix = np.load(args.input_error)

    
    def update_tolerance(val):
        tolerance = 10**slider_tolerance.val - 1

        ax.clear()

        Bezier.get_knots(tolerance)

        points_from_knots_bezier = []

        for i in range(len(Bezier.final_knots_idx) - 1):
            idx = Bezier.final_knots_idx[i]
            next_idx = Bezier.final_knots_idx[i + 1]

            gradient_left = Bezier.tangent_points[idx]
            gradient_right = Bezier.tangent_points[next_idx]

            if np.array_equal(gradient_left, np.array([0, 0])) and np.array_equal(gradient_right, np.array([0, 0])):
                curve_bezier, _ = Bezier.fit_directly_bezier(Bezier.points[idx:next_idx + 1], steps, no_fixed_endpoints=False, return_T=True)
            elif np.array_equal(gradient_left, np.array([0, 0])):
                curve_bezier, _ = Bezier.fit_fixed_bezier_single_gradient(Bezier.points[idx:next_idx + 1], steps, gradient_right, use_left=False)
            elif np.array_equal(gradient_right, np.array([0, 0])):
                curve_bezier, _ = Bezier.fit_fixed_bezier_single_gradient(Bezier.points[idx:next_idx + 1], steps, gradient_left, use_left=True)
            else:
                curve_bezier, _ = Bezier.fit_fixed_bezier(Bezier.points[idx:next_idx + 1], steps, gradient_left, gradient_right)

            ax.scatter(curve_bezier[1:3, 0], curve_bezier[1:3, 1], s=1, color='gray', alpha=0.5)
            curve_points, _ = Bezier.extract_points_bezier(curve_bezier, num_points)
            points_from_knots_bezier.append(curve_points)

        points_from_knots_bezier = np.concatenate(points_from_knots_bezier, axis=0)

        ax.plot(X, Y, label='Dados Originais')

        for i in Bezier.final_knots_idx:
            x, y = Bezier.points[i]
            dx, dy = Bezier.tangent_points[i]
            ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='blue', label='Tangente' if i == 0 else "")

        ax.set_title(f'ε = {args.epsilon:.2f}')
        ax.plot(points_from_knots_bezier[:, 0], points_from_knots_bezier[:, 1], label='Pontos da curva', color='red')
        ax.plot(Bezier.final_knots[:, 0], Bezier.final_knots[:, 1],'o', label='Pontos de controle', color='orange')
        print(f"τ = {tolerance:.2f}")

        fig.canvas.draw_idle()

    slider_tolerance.on_changed(update_tolerance)

    plt.show()
        


if "__main__" == __name__:

    main()
