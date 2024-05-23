from pathlib import Path

import numpy as np
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def early_exit(message):
    print(message)
    exit()


def plot_combined(name, results, change_steps):
    results = np.array(results)
    print(results.shape)
    xs = np.arange(results.shape[1])
    ys = np.mean(results, axis=0)
    yerrs = stats.sem(results, axis=0)
    plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
    plt.plot(xs, ys, label=name)
    for step in change_steps:
        plt.axvline(x=step, color='r', linestyle='--', label='Env Change' if step == change_steps[0] else "")



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        default=None,  # "results"
        help="Directory containing the results of the runs",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path in which to save the output image"
    )
    parser.add_argument(
        "--seeds", required=True, help="Comma-separated list of seeds to plot"
    )
    parser.add_argument("--type", required=True, help="Type of parameter changed", default="geom_friction")

    args = parser.parse_args()

    if args.seeds.isdigit():
        seeds = [int(args.seeds)]
    else:
        seeds = [int(seed) for seed in args.seeds.split(",")]

    all_results = {"No early termination": []}
    if args.directory:
        directory = Path(args.directory)
        if not directory.is_dir():
            early_exit(f"{directory.resolve()} is not a directory")
    
    for seed in seeds:
        if directory is not None:
            format_str = f"Hopper-v4-env={{}}-seed={seed}"
            all_results["No early termination"].append(np.load(directory / format_str.format(args.type) / "scores.npy"))

    change_steps = np.arange(20, 201, 20)

    plt.figure()
    plt.title("Hopper-v4 " + args.type)
    plt.xlabel("Iteration")
    plt.ylabel("Average Return")
    for name, results in all_results.items():
        plot_combined(name, results, change_steps)
    plt.legend()
    plt.savefig(Path(args.type + args.output), bbox_inches="tight")
