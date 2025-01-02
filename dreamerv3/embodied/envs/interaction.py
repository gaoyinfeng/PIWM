import sys
import pathlib
import gym
import numpy as np
import embodied

# TODO: client_interface should in gym folder?
from interaction_dreamerv3.client_interface import ClientInterface

# isim env
class Interaction(embodied.Env):
  
  def __init__(self, task, args):
    self._task = task
    self._args = args
    self._env = ClientInterface(self._args)
    self._done = True
    print('Set I-SIM env successfully!')

  @property
  def obs_space(self):
    # for prediction decode target, needs to separate different vehicles and predict each of them
    if self._task == 'prediction':
      # ego observation space
      obs_space = {
          'ego': embodied.Space(np.float64, (19, 5)),
          'ego_prediction': embodied.Space(np.float64, (self._args['predict_horizen'], 2)),
          # 'ego_map': embodied.Space(np.float64, shape),
      }
      # plus vdi observation space
      for i in range(self._args['vdi_num']):
        obs_space[f'vdi_{i+1}'] = embodied.Space(np.float64, (19, 5))
        obs_space[f'vdi_{i+1}_prediction'] = embodied.Space(np.float64, (self._args['predict_horizen'], 2))
      obs_space.update({
                        'id_vdi': embodied.Space(np.int32, (self._args['vdi_num'])),
                        'mask_vdi': embodied.Space(np.int32, (self._args['vdi_num'])),
                        'should_init_vdi': embodied.Space(np.int32, (self._args['vdi_num'])),
                        })
      # plus vpi(vdi with no interaction / is too away from ego) observation space
      for i in range(self._args['vpi_num']):
        obs_space[f'vpi_{i+1}'] = embodied.Space(np.float64, (19, 5))
      obs_space.update({
                        'mask_vpi': embodied.Space(np.int32, (self._args['vpi_num'])),
                        'should_init_vpi': embodied.Space(np.int32, (self._args['vpi_num'])),
                        })
      
    # for PIM with only branch structure network, we need to recon every vehicle's state
    elif self._task == 'branch':
      obs_space = {'ego': embodied.Space(np.float64, (19, 5))}
      # plus vdi observation space
      for i in range(self._args['vdi_num']):
        obs_space[f'vdi_{i+1}'] = embodied.Space(np.float64, (19, 5))
      obs_space.update({
                        'id_vdi': embodied.Space(np.int32, (self._args['vdi_num'])),
                        'mask_vdi': embodied.Space(np.int32, (self._args['vdi_num'])),
                        'should_init_vdi': embodied.Space(np.int32, (self._args['vdi_num'])),
                        })
      # plus vpi(vdi with no interaction / is too away from ego) observation space
      for i in range(self._args['vpi_num']):
        obs_space[f'vpi_{i+1}'] = embodied.Space(np.float64, (19, 5))
      obs_space.update({
                        'mask_vpi': embodied.Space(np.int32, (self._args['vpi_num'])),
                        'should_init_vpi': embodied.Space(np.int32, (self._args['vpi_num'])),
                        })

    # for reconstraction decode target, we treat all vehicles as one whole state
    elif self._task == 'recon':
      # different state sizes
      ego_state_size = (19, 5)
      vdi_state_size = (19, 5)
      vpi_state_size = (19, 5)
      # full state size
      full_state_size = np.prod(ego_state_size) + np.prod(vdi_state_size) * self._args['vdi_num'] + np.prod(vpi_state_size) * self._args['vpi_num'] 
      # ego observation space
      obs_space = {
          'state': embodied.Space(np.float64, (int(full_state_size))),
      }

    # plus episode observation space
    obs_space.update({
                      'reward': embodied.Space(np.float32),
                      'is_first': embodied.Space(bool),
                      'is_last': embodied.Space(bool),
                      'is_terminal': embodied.Space(bool),
    })
    # plus vpi useful statistics observation space
    obs_space.update({
                      'sta_speed': embodied.Space(np.float32),
                      'sta_collision': embodied.Space(np.int32),
                      'sta_success': embodied.Space(np.int32),
                      'sta_complet': embodied.Space(np.float32),
                      'sta_gt_distance': embodied.Space(np.float32),
    })
    return obs_space

  @property
  def act_space(self):
    return {'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
            }

  @property
  def state_frame(self):
    return self._args['state_frame']

  def step(self, action):
    # reset the environment
    if self._done:
      self._done = False
      state_dict = self._env.reset()
      # TODO: we only consider one ego vehicle for now
      for ego_id in state_dict.keys():
        state = state_dict[ego_id]

      # prediction decode target, separate vehicle state branch
      if self._task == 'prediction':
        # ego obs
        obs = {
          'ego': state['ego'],
          # 'ego_map': state['ego_map'],
          }
        # vdi obs
        for i in range(self._args['vdi_num']):
          obs.update({f'vdi_{i+1}': state[f'vdi_{i+1}']})
          # obs.update({f'vdi_{i+1}_map': state[f'vdi_{i+1}']})
        obs.update({
                    'id_vdi': state['id_vdi'],
                    'mask_vdi': state['mask_vdi'],
                    'should_init_vdi': state['should_init_vdi'],
                    })
        # vpi obs
        for i in range(self._args['vpi_num']):
          obs.update({f'vpi_{i+1}': state[f'vpi_{i+1}']})
        obs.update({
                    'mask_vpi': state['mask_vpi'],
                    'should_init_vpi': state['should_init_vpi'],
                    })
        
      # branch decode target, separate vehicle state branch
      elif self._task == 'branch':
        # ego obs
        obs = {'ego': state['ego']}
        # vdi obs
        for i in range(self._args['vdi_num']):
          obs.update({f'vdi_{i+1}': state[f'vdi_{i+1}']})
        obs.update({
                    'id_vdi': state['id_vdi'],
                    'mask_vdi': state['mask_vdi'],
                    'should_init_vdi': state['should_init_vdi'],
                    })
        for i in range(self._args['vpi_num']):
          obs.update({f'vpi_{i+1}': state[f'vpi_{i+1}']})
        obs.update({
                    'mask_vpi': state['mask_vpi'],
                    'should_init_vpi': state['should_init_vpi'],
                    })

      # recon decode target, concat all vehicles states as one(by their order, from cloest to farest, zero padding)
      elif self._task == 'recon':
        order = state['index_vdi'] + state['index_vpi']
        # for zero-padding vdi and vpi
        for i in range(self._args['vdi_num']):
          if f'vdi_{i+1}' not in order:
            order.append(f'vdi_{i+1}')
        for i in range(self._args['vpi_num']):
          if f'vpi_{i+1}' not in order:
            order.append(f'vpi_{i+1}')
        # concat ego, vdis and vpis features, vdis and vpis features are ordered by distance
        value = state['ego'].reshape((-1))
        for key_order in order:
          value = np.concatenate([value, state[key_order].reshape(-1)], axis = 0)
        obs = {'state': value}

      # episode obs
      obs.update({
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
      })
      # statistics obs
      obs.update({
        'sta_speed': 0.0,
        'sta_collision': 0,
        'sta_success': 0,
        'sta_complet': 0,
        'sta_gt_distance': 0,
      })
      return obs

    # step the environment
    # TODO: consider multiple ego vehicles
    action_dict = {self._env.ego_id_list[0]: [action['action']]}
    prediction = action['prediction'] if 'prediction' in action.keys() else None
    state_dict, reward_dict, done_dict, aux_info_dict = self._env.step(action_dict, prediction=prediction)
    # TODO: we only consider one ego vehicle for now
    for ego_id in state_dict.keys():
      state = state_dict[ego_id]
      reward = reward_dict[ego_id]
      self._done = done_dict[ego_id]
      aux_info = aux_info_dict[ego_id]

    # prediction decode target, separate vehicle state branch
    if self._task == 'prediction':
      # ego obs
      obs = {'ego': state['ego'],
            # 'ego_map': state['ego_map'],
            }
      # vdi obs
      for i in range(self._args['vdi_num']):
        obs.update({f'vdi_{i+1}': state[f'vdi_{i+1}']})
        # obs.update({f'vdi_{i+1}_map': state[f'vdi_{i+1}']})
      obs.update({
                  'id_vdi': state['id_vdi'],
                  'mask_vdi': state['mask_vdi'],
                  'should_init_vdi': state['should_init_vdi'],
                  })
      # vpi obs
      for i in range(self._args['vpi_num']):
        obs.update({f'vpi_{i+1}': state[f'vpi_{i+1}']})
      obs.update({
                  'mask_vpi': state['mask_vpi'],
                  'should_init_vpi': state['should_init_vpi'],
                  })
      
    # branch decode target, separate vehicle state branch
    elif self._task == 'branch':
      # ego obs
      obs = {'ego': state['ego']}
      # vdi obs
      for i in range(self._args['vdi_num']):
        obs.update({f'vdi_{i+1}': state[f'vdi_{i+1}']})
      obs.update({
                  'id_vdi': state['id_vdi'],
                  'mask_vdi': state['mask_vdi'],
                  'should_init_vdi': state['should_init_vdi'],
                  })
      for i in range(self._args['vpi_num']):
        obs.update({f'vpi_{i+1}': state[f'vpi_{i+1}']})
      obs.update({
                  'mask_vpi': state['mask_vpi'],
                  'should_init_vpi': state['should_init_vpi'],
                  })
      
    # recon decode target, concat all vehicles states as one(by their order, from cloest to farest, zero padding)
    elif self._task == 'recon':
      order = state['index_vdi'] + state['index_vpi']
      # for zero-padding vdi and vpi
      for i in range(self._args['vdi_num']):
        if f'vdi_{i+1}' not in order:
          order.append(f'vdi_{i+1}')
      for i in range(self._args['vpi_num']):
        if f'vpi_{i+1}' not in order:
          order.append(f'vpi_{i+1}')
      # concat ego and vdis features, vdi feature is ordered by distance
      value = state['ego'].reshape((-1))
      for key_order in order:
        value = np.concatenate([value, state[key_order].reshape(-1)], axis=0)
      obs = {'state': value}
    
    # episode obs
    obs.update({
        'reward': reward,
        'is_first': False,
        'is_last': self._done,
        'is_terminal': self._done,
    })
    # statistics obs
    obs.update({
      'sta_speed': aux_info['speed'],
      'sta_collision': aux_info['result'] == 'collision',
      'sta_success': aux_info['result'] == 'success',
      'sta_complet': aux_info['completion_rate'],
      'sta_gt_distance': aux_info['distance_to_gt'],
    })
    
    return obs
