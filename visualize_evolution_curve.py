import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

from src.bezierCurveFitter import BezierCurveFitter

def main():
    parser = argparse.ArgumentParser(description='Visualize the evolution of a Bezier curve fitting process.')
    parser.add_argument('--window_size', type=int, default=10, help='Size of the sliding window for tangent calculation.')
    parser.add_argument('--num_steps', type=int, default=1, help='Number of steps for the fitting process.')
    parser.add_argument('--input_file', type=str, default='examples/3-trace.txt', help='Path to the input file containing points.')
    parser.add_argument('--output_file', type=str, default='RDP_epsilon10.gif', help='Path to save the output animation.')
    parser.add_argument('--epsilon', type=int, default=2, help='Epsilon value for the RDP algorithm.')
    args = parser.parse_args()


    # Pegar os pontos
    points = np.loadtxt(args.input_file, delimiter=' ')  # Adjust delimiter if needed
    X = points[:, 0]
    Y = points[:, 1]

    P = np.column_stack((X, Y))


    # --- Código inicial ---
    Bezier = BezierCurveFitter(P)

    knots = Bezier.rdp(P, args.epsilon, [0, len(P) - 1])
    corners = Bezier.get_corners(P, knots)

    _, tangent_points = Bezier.sliding_window(knots, corners, args.num_steps, window_size=args.window_size)

    fig, ax = plt.subplots()
    ax.plot(P[:, 0], P[:, 1], label='Dados Originais')

    for i, (x, y) in enumerate(zip(X, Y)):
        if i < len(tangent_points):  
            dx, dy = tangent_points[i]
            ax.quiver(x, y, dy, -dx, angles='xy', scale_units='xy', scale=1, color='blue', label='Tangente' if i == 0 else "")

    ax.legend()
    ax.set_title('Curva Ajustada')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.axis('equal')
    plt.show()

    fig, ax = plt.subplots()

    x_min, x_max = np.min(X) - 10, np.max(X) + 10
    y_min, y_max = np.min(Y) - 10, np.max(Y) + 10
    ax.set_xlim(x_min - 1, x_max + 1)  
    ax.set_ylim(y_min - 1, y_max + 1)  
    ax.set_aspect('equal')

    def update(frame):
        ax.clear()

        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_aspect('equal')

        epsilon = (2* (frame + 1)) 
        steps = args.num_steps
        num_points = 50

        knots, indices = Bezier.get_knots(P, tangent_points, epsilon, steps)
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
        ax.set_title(f'ε = {epsilon:.2f}')
        ax.scatter(knots[:, 0], knots[:, 1], label='Pontos de controle', color='orange')
        ax.plot(points_from_knots_bezier[:, 0], points_from_knots_bezier[:, 1], label='Pontos da curva', color='red')
        print(epsilon)

    anim = animation.FuncAnimation(fig, update, frames=range(9, -1, -2), interval=1500)
    anim.save(args.output_file, writer='pillow')




        


if "__main__" == __name__:

    main()
