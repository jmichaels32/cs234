import argparse
import pathlib
import time

try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import stable_baselines3 as sb3
import matplotlib.pyplot as plt

from util import export_plot

class FrictionWrapper(gym.Wrapper):
    def __init__(self, env, initial_friction, friction_increment):
        super().__init__(env)
        self.friction = initial_friction
        self.friction_increment = friction_increment
        self.step_count = 0
        self.actual_step_count = 0
        self.friction_values = []

    def step(self, action):
        # Adjust the friction parameters
        self._adjust_friction()
        
        # Perform the step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update step count
        self.step_count += 1
        if self.step_count % 1000000 == 0:
            self.actual_step_count += 1
        
        return obs, reward, terminated, truncated, info

    def _adjust_friction(self):
        # Gradually increase friction
        new_friction = self.friction + self.friction_increment * self.actual_step_count
        for i in range(len(self.env.model.geom_friction)):
            self.env.model.geom_friction[i] = new_friction
        self.friction_values.append(new_friction)


def evaluate(env, policy):
    model_return = 0
    T = env.spec.max_episode_steps
    obs, _ = env.reset()
    for _ in range(T):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        model_return += reward
        if done:
            break
    return model_return

class EvalCallback(sb3.common.callbacks.BaseCallback):
    def __init__(self, eval_period, num_episodes, env, policy):
        super().__init__()
        self.eval_period = eval_period
        self.num_episodes = num_episodes
        self.env = env
        self.policy = policy

        self.returns = []

    def _on_step(self):
        if self.n_calls % self.eval_period == 0:
            print(f"Evaluating after {self.n_calls} steps")
            model_returns = []
            for _ in range(self.num_episodes):
                model_returns.append(evaluate(self.env, self.policy))
            self.returns.append(np.mean(model_returns))

        # If the callback returns False, training is aborted early.
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl-steps",
        type=int,
        help="The number of learning iterations",
        default=1000000,
    )
    parser.add_argument(
        "--early-termination", help="Terminate the episode early", action="store_true"
    )
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()

    output_path = pathlib.Path(__file__).parent.joinpath(
        "results",
        f"Hopper-v4-early-termination={args.early_termination}-seed={args.seed}",
    )
    model_output = output_path.joinpath("model.zip")
    log_path = output_path.joinpath("log.txt")
    scores_output = output_path.joinpath("scores.npy")
    plot_output = output_path.joinpath("scores.png")
    friction_plot_output = output_path.joinpath("friction.png")

    env = gym.make(
        "Hopper-v4",
        terminate_when_unhealthy=args.early_termination,
    )

    # Wrapping the environment to modify friction over time
    env = FrictionWrapper(env, initial_friction=0.5, friction_increment=10)

    agent = sb3.PPO("MlpPolicy", env, verbose=1)
    eval_callback = EvalCallback(
        args.rl_steps // 100,
        10,
        env,
        lambda obs: agent.predict(obs)[0],
    )
    start = time.perf_counter()
    agent.learn(args.rl_steps, callback=eval_callback)
    end = time.perf_counter()

    # Log the results
    returns = eval_callback.returns
    if not output_path.exists():
        output_path.mkdir(parents=True)
    agent.save(model_output)
    with open(log_path, "w") as f:
        f.write(f"Wall time elapsed: {end-start:.2f}s\n")
    np.save(scores_output, returns)
    export_plot(returns, "Returns", "Hopper-v4", plot_output)

    # Plot friction values over time
    plt.figure()
    plt.plot(env.friction_values)
    plt.xlabel("Steps")
    plt.ylabel("Friction")
    plt.title("Friction Over Time")
    plt.savefig(friction_plot_output)
    plt.close()