import torch
from torch import nn
from torch.optim import AdamW
import gymnasium as gym
import pygame
import numpy as np

def cosine_similarity(vec1, vec2):
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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
      direction_reward_scale: float = 0.5, # halve the time penalty if agent is moving directly towards the target
      edge_penalty: float = -20,
      max_steps: int = 1800,
      render_mode = 'human',
  ):
    self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    self.size = size                                     # Size of space in pixels
    self.agent_size = agent_size                         # Size of agent's circular collision shape
    self.friction = friction                             # Coefficient of friction
    self.accel = accel                                   # Acceleration of movement, measured in px/frame^2
    self.min_target_dist_coeff = min_target_dist_coeff   # Relates environment size to the min distance for agent to travel
    self.goal_reward = goal_reward                       # Reward for reaching goal
    self.time_penalty = time_penalty                     # Reward lost with each frame
    self.direction_reward_scale = direction_reward_scale # Scale of the reward depending on what direction the agent is moving
    self.edge_penalty = edge_penalty                     # Reward lost for hitting edge
    self.max_steps = max_steps                           # max number of steps before an episode is truncated; passed to the TimeLimit wrapper

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
      # use cosine similarity to scale down the time penalty if the agent is moving towards the target
      # conversely, the penalty will be increased if the agent is moving in the opposite direction
      penalty = self.time_penalty
      penalty *= self.direction_reward_scale ** -cosine_similarity(
        self._target_position - self._agent_position,
        self._agent_v
      )
      reward += penalty
    
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
  
  
  def setup_pygame(self):
    if self.window is None and self.render_mode == 'human':
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(
        (self.window_size, self.window_size)
      )
    if self.clock is None and self.render_mode == 'human':
      self.clock = pygame.time.Clock()

  def cleanup_pygame(self):
    if (self.window is not None):
      pygame.display.quit()
      self.window = None
    if (self.clock is not None):
      self.clock = None


  def _render_canvas(self, render_to_window = True):
    if render_to_window:
      self.setup_pygame()
    else:
      self.cleanup_pygame()

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

    if self.render_mode == 'human' and self.window is not None:
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()

      self.clock.tick(self.metadata['render_fps'])
    else:
      # swap the width and height axes with transpose
      # pixels3d returns shape (width, height, 3)
      return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
  
  def render(self, render_to_window = True):
    self._render_canvas(render_to_window)

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

softmax = nn.Softmax(dim=1)


env = SlipperyEnv(
  size=500,
  agent_size=30,
  min_target_dist_coeff=0.2,

  friction=0.95,
  accel=0.05,

  goal_reward=20,
  time_penalty=-1/60,
  direction_reward_scale=0.3,
  edge_penalty=-20,

  max_steps=3600,
)

env = gym.wrappers.TimeLimit(env, env.max_steps)
unwrapped = env.unwrapped

policy = SlipperyAgentNN()
policy.to(device)
# learning rate for policy network
a_theta = 0.01
policy_optim = AdamW(policy.parameters(), lr=a_theta)

baseline = SlipperyRewardEstNN()
baseline.to(device)
# learning rate for baseline network
a_b = 0.01
baseline_optim = AdamW(baseline.parameters(), lr=a_b)

gamma = 0.95
return_scale = 1

epsilon = 0.2
epsilon_decay = 0.9
min_epsilon = 0.01

observation, info = env.reset()
norm_obs = torch.Tensor(unwrapped.normalize_obs(observation))
norm_obs = norm_obs.to(device) # move to gpu
episode = 0

while True:
  total_reward = 0
  observation, info = env.reset()

  # history for states, actions, rewards, discounted rewards, and the probability of the chosen action
  s = []
  a = []
  r = []
  dr = []
  pi = []

  episode_length = 0
  #should_render = episode % 100 == 99
  should_render = False
  
  # run the episode
  while True:
    norm_obs = torch.tensor(unwrapped.normalize_obs(observation), dtype=torch.float32, device=device)
    # get action
    action = None
    
    policy_logits = policy.forward(norm_obs)
    action_probabilities = torch.softmax(policy_logits, dim=-1)
    # pick a random action with probability `epsilon`
    if (env.np_random.random() < epsilon):
      action = env.np_random.integers(0, 8)
    else:
      # greedy
      action = torch.argmax(action_probabilities).item()

    s.append(norm_obs)
    a.append(action)
    pi.append(action_probabilities[action])
    observation, reward, terminated, truncated, info = env.step(action)
    r.append(reward)
    total_reward += reward

    episode_length += 1

    if (should_render):
      unwrapped.render()

    # check if the episode is over
    if terminated or truncated:
      # clean up pygame if this episode was rendered
      if (should_render):
        unwrapped.cleanup_pygame()
      break
  
  #----------------------
  # agent network updates
  #----------------------

  # calculate discounted rewards and train baseline network
  dr = [0] * episode_length
  dr[episode_length - 1] = r[-1]
  baseline_loss = 0
  for i in range(episode_length - 2, -1, -1):
    dr[i] = r[i] + gamma * dr[i + 1] * return_scale

    # use squared error
    baseline_loss += (baseline.forward(s[i])[0] - dr[i]) ** 2

  baseline.zero_grad()
  baseline_loss.backward()
  baseline_optim.step()
  
  # train policy
  policy_loss = 0
  for i in range(episode_length):
    # get separate baseline r value, detached from baseline's computation graph
    b_value = baseline.forward(s[i])[0].detach()
    policy_loss += -torch.log(pi[i]) * (dr[i] - b_value)
  policy.zero_grad()
  policy_loss.backward()
  policy_optim.step()

  print(episode, np.mean(r), baseline_loss.item(), policy_loss.item())
  episode += 1