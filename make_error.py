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
    parser.add_argument('--output_error', type=str, default='errors/error_g.npy', help='Path to save the output error.')
    parser.add_argument('--epsilon', type=float, default=2, help='Epsilon value for the RDP algorithm.')

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

    Bezier.sliding_window(args.num_steps, window_size=args.window_size)

    Bezier.make_error_matrix(args.num_steps)

    np.save(args.output_error, Bezier.error_matrix)
        


if "__main__" == __name__:

    main()