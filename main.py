import torch
import gymnasium as gym
import pygame as pg
import numpy as np

class SlipperyEnv(gym.Env):
  def __init__(
      self,
      size: int = 500,
      agent_size: int = 20,
      friction: float = 0.95,
      accel_coeff = 0.0001,
      min_target_dist_coeff: float = 0.4,
  ):
    self.size = size                                    # Size of space in pixels
    self.agent_size = agent_size                        # Size of agent's circular collision shape
    self.friction = friction                            # Coefficient of friction
    self.accel_coeff = accel_coeff                      # Strength of movement actions, relative to the size of the space
    self.min_target_dist_coeff = min_target_dist_coeff  # Relates environment size to the min distance for agent to travel
    if (self.min_target_dist_coeff >= 0.5):
      raise ValueError('min_target_dist_coeff must be less than 0.5')

    self.min_target_dist = self.size * self.min_target_dist_coeff

    self._agent_position = np.array([0, 0], dtype=float)
    self._target_position = np.array([0, 0], dtype=float)

    self._agent_v = np.array([0, 0], dtype=float)

    # Continuous, bounded coordinate space for both the agent and the target
    self.observation_space = gym.spaces.Dict({
      'agent': gym.spaces.Box(0, size - 1, shape=(2,), dtype=float),
      'target': gym.spaces.Box(0, size - 1, shape=(2,), dtype=float)
    })

    self.action_space = gym.spaces.Discrete(4)

    self.accel = self.size * self.accel_coeff
    self._action_to_dv = {
      0: np.array([0, 0]),
      1: np.array([0, self.accel]),
      2: np.array([self.accel, 0]),
      3: np.array([0, -self.accel]),
      4: np.array([-self.accel, 0]),
    }


  def _get_obs(self):
    """Gets observation for .reset() and .step()
    Returns:
      dict: observation of current environment (agent and target)
    """
    return {'agent': self._agent_position, 'target': self._target_position}
  
  def _get_info(self):
    return None


  def _get_dist_to_target(self):
    """Returns distance between agent and target
      Returns:
        float: distance
    """
    return np.linalg.norm(self._agent_position - self._target_position)


  def _reset_target_position(self):
    """Moves target to a random valid position based on the position of agent
    Returns:
      np.array: new target location
    """
    while True:
      self._target_position = self.np_random.random(size=(2,)) * self.size
      if self._get_dist_to_target() >= self.min_target_dist:
        break
      
    return self._target_position
  
  def _respawn_agent(self):
    """Moves agent to a random valid position based on the position of target and sets its velocity to zero
    Returns:
      np.array: new agent location
    """
    while True:
      self._agent_position = self.np_random.random(size=(2,)) * self.size
      if self._get_dist_to_target() >= self.min_target_dist:
        break
    # set velo to 0
    self._agent_v[:] = 0.0

    return self._agent_position


  def reset(self, seed: int | None = None, options: dict | None = None):
    """Starts a new episode.
    Args:
      seed: RNG seed for reproducible episodes
      options: additional config
    Returns:
      tuple: (observation, info) for initial state
    """

    # seed RNG
    super().reset(seed=seed)

    self._agent_position = self.np_random.random(size=(2,)) * self.size
    self._reset_target_position()

    self._agent_v = np.array([0, 0], dtype=float)

    observation = self._get_obs()
    info = self._get_info()

    return observation, info

  def step(self, action):
    """Execute an action within the environment.
    Args:
      action: The action to take
    
    Returns:
      tuple: (observation, reward, terminated, truncated, info)
    """
    # accelerate the agent according to the action
    dv = self._action_to_dv[action]
    self._agent_v += dv

    # update the agent's position based on its velocity
    self._agent_position += self._agent_v

    # reward
    reached = False
    reward = 0
    if self._get_dist_to_target() < self.agent_size:
      reached = True
      reward += 1
      self._reset_target_position
    else:
      reward += -0.01
    
    pos = self._agent_position
    if (
      pos[0] < 0 or
      pos[0] > self.size or
      pos[1] < 0 or
      pos[1] > self.size
    ):
      # heavy penalty for "falling off the edge"
      reward += -3
      # respawn agent
      self._respawn_agent()
    
    observation = self._get_obs()
    terminated = False
    truncated = False
    info = self._get_info()
    return observation, reward, terminated, truncated, info



    

env = SlipperyEnv()

print(env.reset())