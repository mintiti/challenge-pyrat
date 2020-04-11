from gym.envs.registration import register

register(
    id='PyRatEnv-v0',
    entry_point='pyrat_env.envs:PyratEnv',
    nondeterministic=False,
    max_episode_steps=2000
)
