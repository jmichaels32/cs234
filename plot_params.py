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

    plt.figure()
    plt.title("Hopper-v4 " + args.type + " Parameters")
    plt.xlabel("Iteration")
    plt.ylabel(args.y_label)

    data = [1, 10, 1, 10, 1, 10, 1, 10, 5, 50]
    step_size = 20  
    data = np.repeat(data, step_size)
    change_steps = np.arange(20, len(data) * step_size + 1, step_size)

    plt.plot(change_steps, data)
    plt.savefig(args.type + "parameters.png")