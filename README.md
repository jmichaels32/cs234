Model Settings Pipeline

self.env -> Overall environment (Hopper-v4, Cheetah, etc.)

self.env.model -> Model struct (mujoco._structs.MjModel)

Important Links

HopperEnv
https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper_v4.py

Hopper XML (defines Hopper object)
https://github.com/openai/gym/blob/b1d645caa983c80591c942821e23875ccc483073/gym/envs/mujoco/assets/hopper.xml#L29

MjModel Struct
https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel

Mid Values:
friction_values = [1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 2.0, 0.5]
margin_values = [0.001, 0.005, 0.001, 0.005, 0.001, 0.005, 0.001, 0.005, 0.01, 0.0005]
mass_values = [4, 5, 4, 5, 4, 5, 4, 5, 6, 3]
gravcomp_values = [1, 2, 1, 2, 1, 2, 1, 2, 3, 0]

Final Values:
friction_values = [(1.0, 5.e-3, 1.e-4), 
                       (10., 5.e-1, 1.e-2), 
                       (1.0, 5.e-3, 1.e-4), 
                       (10., 5.e-1, 1.e-2), 
                       (1.0, 5.e-3, 1.e-4), 
                       (10., 5.e-1, 1.e-2), 
                       (1.0, 5.e-3, 1.e-4), 
                       (10., 5.e-1, 1.e-2), 
                       (5.0, 5.e-2, 1.e-3), 
                       (20., 5., 1.e-1)]
margin_values = [0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.01, 1]
mass_values = [4, 16, 4, 16, 4, 16, 4, 16, 8, 32]
gravcomp_values = [1, 10, 1, 10, 1, 10, 1, 10, 5, 50]