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

class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, change_index, geom_friction=-1, geom_margin=-1, body_mass=-1, body_gravcomp=-1):
        super().__init__(env)
        self.geom_friction = geom_friction
        self.geom_margin = geom_margin
        self.body_mass = body_mass
        self.body_gravcomp = body_gravcomp

        self.index = 0
        self.change_index = change_index

    def step(self, action):
        self._adjust(self.geom_friction, 'geom_friction')
        self._adjust(self.geom_margin, 'geom_margin')
        self._adjust(self.body_mass, 'body_mass')
        self._adjust(self.body_gravcomp, 'body_gravcomp')

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.env._elapsed_steps % self.change_index == 0:
            self.index += 1
        
        return obs, reward, terminated, truncated, info

    def _adjust(self, values, attribute):
        if values == -1:
            return
        
        for i in range(len(getattr(self.env.model, attribute))):
            if attribute == 'body_mass' and i == 0:
                continue

            if attribute == 'geom_friction':
                getattr(self.env.model, attribute)[i][0] = values[self.index]
            else: 
                getattr(self.env.model, attribute)[i] = values[self.index]
        
'''
class FrictionWrapper(gym.Wrapper):
    def __init__(self, env, initial_friction, friction_increment):
        super().__init__(env)
        self.friction = initial_friction
        self.friction_increment = friction_increment
        self.actual_step_count = 0
        self.friction_values = []
        self.friction_iteration_values = [self.friction + self.friction_increment * i for i in range(1, 3)]

    def step(self, action):
        # Adjust the friction parameters
        self._adjust_friction()
        
        # Perform the step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update step count
        if self.env._elapsed_steps % 250000 == 0:
            self.actual_step_count += 1
        
        return obs, reward, terminated, truncated, info

    def _adjust_friction(self):
        # Gradually increase friction
        friction_index = self.actual_step_count % 2
        new_friction = self.friction_iteration_values[friction_index]
        #new_friction = self.friction + self.friction_increment * self.actual_step_count
        print(self.env.__dict__)
        print(self.env)
        print(dir(self.env))
        print(self.env._elapsed_steps)
        print(type(self.env.model))
        print(dir(self.env.model))
        print(self.env.model.geom_friction)
        print(self.env.model.body_mass)
        print(self.env.model.tendon_stiffness)
        print(self.env.model.body_gravcomp)
        print(self.env.model.geom_margin)
        print("")
        #for i in range(len(self.env.model.geom_friction)):
            #self.env.model.geom_friction[i] = new_friction
        #self.friction_values.append(new_friction)
'''

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
    total_iterations = 1000000
    buckets = 10

    friction_values = [1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 2.0, 0.5]
    margin_values = [0.001, 0.005, 0.001, 0.005, 0.001, 0.005, 0.001, 0.005, 0.01, 0.0005]
    mass_values = [4, 5, 4, 5, 4, 5, 4, 5, 6, 3]
    gravcomp_values = [1, 2, 1, 2, 1, 2, 1, 2, 3, 0]

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

    parser.add_argument("--geom-friction", type=bool, default=False)
    parser.add_argument("--geom-margin", type=bool, default=False)
    parser.add_argument("--body-mass", type=bool, default=False)
    parser.add_argument("--body-gravcomp", type=bool, default=False)

    args = parser.parse_args()

    name = "geom_friction"
    if args.geom_margin:
        name = "geom_margin"
    elif args.body_mass:
        name = "body_mass"
    elif args.body_gravcomp:
        name = "body_gravcomp"

    output_path = pathlib.Path(__file__).parent.joinpath(
        "results",
        f"Hopper-v4-env={name}-seed={args.seed}",
    )
    model_output = output_path.joinpath("model.zip")
    log_path = output_path.joinpath("log.txt")
    scores_output = output_path.joinpath("scores.npy")
    plot_output = output_path.joinpath("scores.png")
    #friction_plot_output = output_path.joinpath("friction.png")

    env = gym.make(
        "Hopper-v4",
        terminate_when_unhealthy=args.early_termination,
    )

    friction_values = friction_values if args.geom_friction else -1
    margin_values = margin_values if args.geom_margin else -1
    mass_values = mass_values if args.body_mass else -1
    gravcomp_values = gravcomp_values if args.body_gravcomp else -1

    # Wrapping the environment to modify friction over time
    env = EnvironmentWrapper(env, total_iterations // 10, geom_friction=friction_values, geom_margin=margin_values, body_mass=mass_values, body_gravcomp=gravcomp_values)

    agent = sb3.PPO("MlpPolicy", env, verbose=1)
    eval_callback = EvalCallback(
        args.rl_steps // 200,
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
    '''
    plt.figure()
    plt.plot(env.friction_values)
    plt.xlabel("Steps")
    plt.ylabel("Friction")
    plt.title("Friction Over Time")
    plt.savefig(friction_plot_output)
    plt.close()
    '''