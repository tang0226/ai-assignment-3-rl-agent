import torch
from torch import nn
import gymnasium as gym
import pygame
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'Using {device} device')

###################
### ENVIRONMENT ###
###################
class SlipperyEnv(gym.Env):
  def __init__(
      self,
      size: int = 500,
      agent_size: int = 20,
      friction: float = 0.95,
      accel = 0.05,
      min_target_dist_coeff: float = 0.4,
      goal_reward: float = 20,
      time_penalty: float = -1/60, # lose 1 reward for every second
      edge_penalty: float = -20,
      max_steps: int = 1800,
      render_mode = 'human',
  ):
    self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    self.size = size                                    # Size of space in pixels
    self.agent_size = agent_size                        # Size of agent's circular collision shape
    self.friction = friction                            # Coefficient of friction
    self.accel = accel                                  # Acceleration of movement, measured in px/frame^2
    self.min_target_dist_coeff = min_target_dist_coeff  # Relates environment size to the min distance for agent to travel
    self.goal_reward = goal_reward                      # Reward for reaching goal
    self.time_penalty = time_penalty                    # Reward lost with each frame
    self.edge_penalty = edge_penalty                    # Reward lost for hitting edge
    self.max_steps = max_steps                          # max number of steps before an episode is truncated; passed to the TimeLimit wrapper

    assert self.friction < 1 and self.friction > 0

    if (self.min_target_dist_coeff >= 0.5):
      raise ValueError('min_target_dist_coeff must be less than 0.5')

    self.min_target_dist = self.size * self.min_target_dist_coeff
    self.max_lateral_speed = self.accel / (1 - self.friction)

    self._agent_position = np.array([0, 0], dtype=float)
    self._target_position = np.array([0, 0], dtype=float)

    self._agent_v = np.array([0, 0], dtype=float)

    # Continuous, bounded coordinate space for both the agent and the target
    self.observation_space = gym.spaces.Dict({
      'agent': gym.spaces.Box(0, size - 1, shape=(2,), dtype=float),
      'target': gym.spaces.Box(0, size - 1, shape=(2,), dtype=float)
    })

    self.action_space = gym.spaces.Discrete(4)

    self._action_to_dv = {
      0: np.array([0, self.accel]),
      1: np.array([self.accel, self.accel]),
      2: np.array([self.accel, 0]),
      3: np.array([self.accel, -self.accel]),
      4: np.array([0, -self.accel]),
      5: np.array([-self.accel, -self.accel]),
      6: np.array([-self.accel, 0]),
      7: np.array([-self.accel, self.accel]),
    }

    self._latest_action = None

    # ensure a valid render_mode value is being passed
    assert render_mode is None or render_mode in self.metadata['render_modes']
    self.render_mode = render_mode

    # used with human render mode
    self.window = None
    self.clock = None

    self.window_size = self.size


  def _get_obs(self):
    """Gets observation for .reset() and .step()
    Returns:
      dict: observation of current environment (agent and target)
    """
    return {'agent': self._agent_position, 'target': self._target_position, 'v': self._agent_v}
  
  def normalize_obs(self, obs):
    """Normalizes values of obs based on environment size and max possible speed
    Args:
      obs: an observation dict returned by .reset() or .step()
    Returns:
      list[float]: a list of normalized observation values
    """
    return [
      obs['agent'][0] / self.size,
      obs['agent'][1] / self.size,
      obs['target'][0] / self.size,
      obs['target'][1] / self.size,
      obs['v'][0] / self.max_lateral_speed,
      obs['v'][1] / self.max_lateral_speed,
    ]

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
    self._latest_action = action

    # accelerate the agent according to the action
    dv = self._action_to_dv[action]
    self._agent_v += dv
    self._agent_v *= self.friction

    # update the agent's position based on its velocity
    self._agent_position += self._agent_v

    # reward
    reward = 0
    if self._get_dist_to_target() < self.agent_size / 2:
      reward += self.goal_reward
      self._reset_target_position()
    else:
      reward += self.time_penalty
    
    pos = self._agent_position
    if (
      pos[0] < 0 or
      pos[0] > self.size or
      pos[1] < 0 or
      pos[1] > self.size
    ):
      # heavy penalty for "falling off the edge"
      reward += self.edge_penalty
      # respawn agent
      self._respawn_agent()
    
    observation = self._get_obs()
    info = self._get_info()
    return observation, reward, False, None, info
  
  
  def _render_canvas(self):
    if self.window is None and self.render_mode == 'human':
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(
        (self.window_size, self.window_size)
      )
    if self.clock is None and self.render_mode == 'human':
      self.clock = pygame.time.Clock()
    
    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((255, 255, 255))
  
    # ice
    pygame.draw.rect(
      canvas,
      (175, 221, 240),
      pygame.Rect(0, 0, self.size, self.size)
    )

    # target
    pygame.draw.circle(
      canvas,
      (0, 170, 0),
      self._target_position,
      5
    )
    # agent
    pygame.draw.circle(
      canvas,
      (0, 0, 0),
      self._agent_position,
      self.agent_size / 2
    )

    if self.render_mode == 'human':
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()

      self.clock.tick(self.metadata['render_fps'])
    else:
      # swap the width and height axes with transpose
      # pixels3d returns shape (width, height, 3)
      return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
  
  def render(self):
    self._render_canvas()

#####################
### AGENT NETWORK ###
#####################
class SlipperyAgentNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(6, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 8),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits

#################################
### REWARD ESTIMATION NETWORK ###
#################################
class SlipperyRewardEstNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(6, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits



env = SlipperyEnv(
  size=500,
  agent_size=30,
  friction=0.995,
  min_target_dist_coeff=0.2,
)

env = gym.wrappers.TimeLimit(env, env.max_steps)
unwrapped = env.unwrapped

agent_network = SlipperyAgentNN()
agent_network.to(device)

est_network = SlipperyRewardEstNN()
est_network.to(device)

observation, info = env.reset()

for episode in range(5):
  total_reward = 0
  observation, info = env.reset()
  while True:
    normalized = torch.Tensor(unwrapped.normalize_obs(observation))
    normalized = normalized.to(device) # move to gpu

    # get action from agent network
    action = torch.argmax(agent_network.forward(normalized)).item()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    if terminated or truncated:
      break
  print(episode, total_reward)