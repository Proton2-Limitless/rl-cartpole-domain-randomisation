"""
Domain-randomisation wrapper for CartPole-v1.

Resamples cart mass, pole mass, and pole length uniformly at random at the
start of every episode (on every reset()), before the underlying env
generates its initial state.

Default CartPole-v1 physics constants, for reference:
    masscart = 1.0
    masspole = 0.1
    length   = 0.5   (half-length of the pole)
"""

import gymnasium as gym
import numpy as np


class CartPoleDRWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        masscart_range=(0.5, 1.5),
        masspole_range=(0.05, 0.2),
        length_range=(0.25, 0.75),
        seed=None,
    ):
        super().__init__(env)
        self.masscart_range = masscart_range
        self.masspole_range = masspole_range
        self.length_range = length_range
        self._rng = np.random.default_rng(seed)

    def _randomize(self):
        u = self.env.unwrapped
        u.masscart = float(self._rng.uniform(*self.masscart_range))
        u.masspole = float(self._rng.uniform(*self.masspole_range))
        u.length = float(self._rng.uniform(*self.length_range))
        # CartPoleEnv caches these as separate attributes derived from the
        # three above — they must be recomputed by hand or the physics step
        # will silently use stale values.
        u.total_mass = u.masspole + u.masscart
        u.polemass_length = u.masspole * u.length

    def reset(self, **kwargs):
        self._randomize()
        return self.env.reset(**kwargs)

    def get_params(self):
        """Convenience getter, useful for logging / debugging."""
        u = self.env.unwrapped
        return {"masscart": u.masscart, "masspole": u.masspole, "length": u.length}


if __name__ == "__main__":
    # Quick sanity check — confirms parameters actually vary across resets.
    # Run this file directly (`python cartpole_dr_wrapper.py`) before
    # touching the training scripts.
    env = CartPoleDRWrapper(gym.make("CartPole-v1"), seed=0)
    for i in range(5):
        env.reset()
        print(f"Episode {i}: {env.get_params()}")
    env.close()
