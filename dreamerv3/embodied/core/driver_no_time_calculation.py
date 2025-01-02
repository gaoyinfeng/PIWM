import collections

import numpy as np

import copy
import time

from .basics import convert
from embodied.envs.interaction_utils import localize_vector_transform_list

class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, envs, **kwargs):
    assert len(envs) > 0
    self._envs = envs
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.reset()

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._envs),) + v.shape, v.dtype))
        for k, v in self._envs.act_space.items()}
    self._acts['reset'] = np.ones(len(self._envs), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._envs))]
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0, predictor=None):
    # if there is a predictor, which means it is i-sim env and use prediction as feature training (decoder in world model) target,
    # and we want to use predictor to online predict future trajectory of surrounding cars
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode, predictor)

  def _step(self, policy, step, episode, predictor=None):
    # reset or step envs (since acts contains both 'action' and 'reset', and 'prediction' for i-sim) 
    assert all(len(v) == len(self._envs) for v in self._acts.values())
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    # step each env, and get new observation
    obs = self._envs.step(acts)
    obs = {k: convert(v) for k, v in obs.items()} # convert data type
    assert all(len(x) == len(self._envs) for x in obs.values()), obs

    # get new actions from policy, 

    # NOTE: for isim env, if state_frame == 'global', then the observation gained from env.step() is in global frame 
    # (to make prediction data), then it will transfered into ego frame for model learning. yet if we want to use this observation 
    # directly with policy and world model, it should be converted into ego local frame first
    obs_for_infer = copy.deepcopy(obs)
    if self._envs.state_frame == 'global':
      for key in obs_for_infer.keys():
        for i in range(len(self._envs)): 
          state = []
          # ego current position and heading
          ego_pos = obs['ego'][i][-1][2:4]
          ego_heading = obs['ego'][i][-1][4]
          # for the vehicles state that are in the scene
          condition = [key.startswith('ego'), key.startswith('vdi_') and obs_for_infer['mask_vdi'][i][int(key[-1])-1], key.startswith('vpi_') and obs_for_infer['mask_vpi'][i][int(key[-1])-1]]
          if sum(condition) > 0:
            # convert position from global frame to ego frame
            localized_traj_state = localize_vector_transform_list(ego_pos, ego_heading, obs_for_infer[key][i])
            state.append(localized_traj_state)
          # for zero padded vehicles obs no needs to convert, for vpi obs(rewards, is_first, is_last, etc.), shouldnt change
          else:
            state.append(obs_for_infer[key][i])
        obs_for_infer[key] = np.array(state)
    
    # make actions
    acts, self._state = policy(obs_for_infer, self._state, **self._kwargs)
    acts = {k: convert(v) for k, v in acts.items()} # convert data type
    # expand actions when some env's episode ends
    if obs['is_last'].any():
      mask = 1 - obs['is_last']
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
    # reset envs on next _step
    acts['reset'] = obs['is_last'].copy()

    # pred_start = time.time()
    # make predictions of surrounding cars if necessary
    if predictor:
      prediction = predictor(obs_for_infer, self._state, **self._kwargs)
      prediction = {k: convert(v) for k, v in prediction.items()} # convert data type
      stacked_prediction = prediction['ego_prediction']
      stacked_prediction = np.expand_dims(stacked_prediction, axis=1)
      # stack prediction as ego, vdi_1 ... vdi_n
      for i in range(len(prediction.keys()) - 1):
        vdi_prediction = np.expand_dims(prediction[f'vdi_{i+1}_prediction'], axis=1)
        stacked_prediction = np.concatenate([stacked_prediction, vdi_prediction], axis=1)
      vdi_mask = obs['mask_vdi']
      ego_mask = np.ones(list(vdi_mask.shape[:1]) + [1])
      mask = np.concatenate([ego_mask, vdi_mask], axis=1)
      mask = mask.reshape(list(mask.shape) + [1,1])
      stacked_prediction = np.where(mask, stacked_prediction, 0.)
      acts['prediction'] = stacked_prediction
    # print('prediction time:', time.time() - pred_start)

    # update acts
    self._acts = acts
    trns = {**obs, **acts}

    # if a new episode begin clear eps data
    if obs['is_first'].any():
      for i, first in enumerate(obs['is_first']):
        if first:
          self._eps[i].clear()

    # save obs and act to self._eps, and call on_step, which is used for Logger to count total interact steps from multiple envs. the input of fn is useless
    for i in range(len(self._envs)):
      trn = {k: v[i] for k, v in trns.items()}
      [self._eps[i][k].append(v) for k, v in trn.items()]
      [fn(trn, i, **self._kwargs) for fn in self._on_steps]
      step += 1

    # if an episode ends, call on_episode
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):
        if done:
          ep = {k: convert(v) for k, v in self._eps[i].items()}
          [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
          episode += 1
    
    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
