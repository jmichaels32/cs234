from pathlib import Path

import numpy as np
import scipy.stats as stats
import matplotlib
from collections import defaultdict

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
    # just plot the perturbed sections once.
    if name == "PPO retrained each epoch":
        plt.fill_between(xs[change_steps[0]:change_steps[1]], ys[change_steps[0]:change_steps[1]] - yerrs[change_steps[0]:change_steps[1]], ys[change_steps[0]:change_steps[1]] + yerrs[change_steps[0]:change_steps[1]], alpha=0.25)
        plt.plot(xs[change_steps[0]:change_steps[1]], ys[change_steps[0]:change_steps[1]], label="PPO retrained each epoch—perturbed")
        plt.fill_between(xs[change_steps[-3]:], ys[change_steps[-3]:] - yerrs[change_steps[-3]:], ys[change_steps[-3]:] + yerrs[change_steps[-3]:], alpha=0.25)
        plt.plot(xs[change_steps[-3]:], ys[change_steps[-3]:], label="PPO retrained each epoch—inference")
    else:
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

    all_results = defaultdict(list)
    if args.directory:
        directory = Path(args.directory)
        if not directory.is_dir():
            early_exit(f"{directory.resolve()} is not a directory")
    
    # adjust for slightly misnamed data folders with the ordering.
    # ppo, ppo with l2 init and 1e-3, ppo with crelu, ppo with both and 1e-2, ppo with 1e-2,
    names = ["PPO with Vanilla MLP", "PPO with CReLU MLP", "PPO with L2 Init MLP ($\lambda=10^{-3}$)", "PPO with L2 Init MLP ($\lambda=10^{-2}$)", "PPO with L2 Init and CReLU MLP ($\lambda=10^{-2}$)",  "PPO retrained each epoch"]
    for i, directory in enumerate([Path("results"), Path("results-reset"), Path("results-deep"), Path("results-crelu"), Path("results-deep-e2-and-crelu"), Path("results-reset-new"), ]):
        for seed in seeds:
            format_str = f"Hopper-v4-env={{}}-seed={seed}"
            all_results[names[i]].append(np.load(directory / format_str.format(args.type) / "scores.npy"))

    change_steps = np.arange(20, 201, 20)
    local_change_steps = []

    plt.figure(dpi=500, figsize=(7, 5))
    plt.title("Hopper-v4 " + args.type)
    plt.xlabel("Iteration")
    plt.ylabel("Average Return")
    for i, (name, results) in enumerate(all_results.items()):
        plot_combined(name, results, local_change_steps)
        if i == len(all_results) - 2:
            local_change_steps = change_steps
    plt.legend(prop={'size': 6})
    plt.savefig(Path(args.type + args.output), bbox_inches="tight")
