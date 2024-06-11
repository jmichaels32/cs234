from pathlib import Path

import numpy as np
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--y-label", required=True)
    parser.add_argument("--type", required=True)
    args = parser.parse_args()

    plt.figure(dpi=500, figsize=(6, 2))
    plt.title("Hopper-v4 " + args.type + " Parameters")
    plt.xlabel("Iteration")
    plt.ylabel(args.y_label)


    friction_values = [1, 10, 1, 10, 1, 10, 1, 10, 5, 20]
    margin_values = [0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.01, 1]
    mass_values = [4, 16, 4, 16, 4, 16, 4, 16, 8, 32]
    gravcomp_values = [1, 10, 1, 10, 1, 10, 1, 10, 5, 50]
    data = eval(args.type)
    step_size = 20  
    data = np.repeat(data, step_size)
    change_steps = np.arange(20, len(data) * step_size + 1, step_size)

    plt.plot(change_steps, data)
    plt.tight_layout()
    plt.savefig(args.type + "parameters.png")