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
    parser.add_argument('--output_file', type=str, default='plot_curves.svg', help='Path to save the output animation.')
    args = parser.parse_args()


    # Pegar os pontos
    points = np.loadtxt(args.input_file, delimiter=' ')  # Adjust delimiter if needed
    X = points[:, 0]
    Y = points[:, 1]

    P = np.column_stack((X, Y))


    # --- Código inicial ---
    Bezier = BezierCurveFitter(P)


    _, tangent_points = Bezier.sliding_window(args.num_steps, window_size=args.window_size)
    
    list_of_epsilons = list(range(2, 50, 5))
    list_number_knots = []
    for epsilon in list_of_epsilons:

        steps = args.num_steps

        knots, _ = Bezier.get_knots(P, tangent_points, epsilon, steps)
        list_number_knots.append(len(knots))

        
        print(epsilon)

    #plot number of knots by epsilon:
    plt.figure(figsize=(10, 6))
    plt.plot(list_of_epsilons, list_number_knots, marker='o', linestyle='-')
    plt.title('Number of knots by epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Number of knots')
    plt.grid()
    plt.savefig(args.output_file, format='svg')
    plt.show()



if "__main__" == __name__:

    main()
