import collections
import datetime
import io
import pathlib
import uuid
import time

import numpy as np
import tensorflow as tf
import embodied

from embodied.envs.interaction_utils import localize_transform_list, localize_vector_transform_list


# replay functions
def count_episodes(directory):
  filenames = list(directory.glob('*.npz'))
  num_episodes = len(filenames)
  num_steps = sum(int(str(n).split('-')[-1][:-4]) - 1 for n in filenames)
  return num_episodes, num_steps

def save_episode(directory, episode):
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  identifier = str(uuid.uuid4().hex)
  length = eplen(episode)
  filename = directory / f'{timestamp}-{identifier}-{length}.npz'
  with io.BytesIO() as f1:
    np.savez_compressed(f1, **episode)
    f1.seek(0)
    with filename.open('wb') as f2:
      f2.write(f1.read())
  return filename


def load_episodes(directory, capacity=None, minlen=1):
  # The returned directory from filenames to episodes is guaranteed to be in temporally sorted order.
  filenames = sorted(directory.glob('*.npz'))
  if capacity:
    num_steps = 0
    num_episodes = 0
    for filename in reversed(filenames):
      length = int(str(filename).split('-')[-1][:-4])
      num_steps += length
      num_episodes += 1
      if num_steps >= capacity:
        break
    filenames = filenames[-num_episodes:]
  episodes = {}
  for filename in filenames:
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode {str(filename)}: {e}')
      continue
    episodes[str(filename)] = episode
  return episodes

def eplen(episode):
  return len(episode['action']) - 1


class ReplayEp:

  def __init__(
      self, directory, capacity, batch_size, batch_length, vdi_num, vpi_num, predict_horizen, 
      ongoing=False, minlen=1, maxlen=0, prioritize_ends=False):
    # make a dir
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(parents=True, exist_ok=True)
    # configs of replay buffer
    self._capacity = capacity
    # NOTE: batch size seems no use in replay dmv3 setting, it is considered 'JAXAgent' class
    self._batch_size = batch_size
    self._batch_length = batch_length
    
    self._ongoing = ongoing # 'False' by defualt in dmv2 configs
    self._minlen = minlen
    self._maxlen = maxlen
    self._prioritize_ends = prioritize_ends # 'True' by defualt in dmv2 configs
    # configs of env
    self._vdi_num = vdi_num
    self._vpi_num = vpi_num
    self._predict_horizen = predict_horizen
    self._predict_task = True if self._predict_horizen > 0 else False

    self._random = np.random.RandomState() # set a np random seed, it needs to be called before every random.random to ensure the outcome are the same
    # filename -> key -> value_sequence
    self._complete_eps = self.load()
    # worker -> key -> value_sequence
    self._ongoing_eps = collections.defaultdict(
        lambda: collections.defaultdict(list))
    self._total_episodes, self._total_steps = count_episodes(directory)
    self._loaded_episodes = len(self._complete_eps)
    self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())

  def __len__(self):
    return self._loaded_steps
  
  @property
  def stats(self):
    return {
        'total_steps': self._total_steps,
        'total_episodes': self._total_episodes,
        'loaded_steps': self._loaded_steps,
        'loaded_episodes': self._loaded_episodes,
    }

  def add(self, transition, worker=0):
    episode = self._ongoing_eps[worker]
    for key, value in transition.items():
      episode[key].append(value)
    if transition['is_last']:
      self.add_episode(episode)
      episode.clear()

  def add_episode(self, true_episode):    
    # episode length consider prediction
    true_length = eplen(true_episode)
    length = true_length - self._predict_horizen

    # only consider data which has prediction label
    if length < self._minlen:
      print(f'Skipping short episode of length {length}.')
      return
    
    episode = {}
    for key, value in true_episode.items():
      episode.update({key: value[:length]})
      if key in ['is_last', 'is_terminal']:
        episode[key][-1] = True

    # make prediction data, and convert data from global frame to ego frame
    ego_state_ego_frame_list = []
    vdi_state_ego_frame_dict = collections.defaultdict(list)
    vpi_state_ego_frame_dict = collections.defaultdict(list) # for vpi vehicles state, we only needs it as state input

    if self._predict_task:
      ego_prediction_ego_frame_list = []
      vdi_prediction_ego_frame_dict = collections.defaultdict(list)

    for time_step in range(len(episode['ego'])):
      # 1. make prediction data in global frame in current time step
      if self._predict_task:
        # ego prediction data in global frame
        ego_prediction = [true_episode['ego'][t][-1][2:4] for t in range(time_step + 1, time_step + self._predict_horizen + 1)] # ...[t][-1][2:4] is the global position in t timestep
        # vdi prediction data in global frame
        vdi_prediction_dict = collections.defaultdict(list)
        for i in range(self._vdi_num):
          
          if episode['mask_vdi'][time_step][i] == 1: # if vdi_i is in the scene in current time step
            # align prediction data with vdi id
            vdi_id = episode['id_vdi'][time_step][i] # get vdi's id
            for t in range(time_step + 1, time_step + self._predict_horizen + 1):
              if true_episode['id_vdi'][t][i] == vdi_id: # if id not change, which means it is the same vdi vehicle
                vdi_prediction_dict[f'vdi_{i+1}'].append(true_episode[f'vdi_{i+1}'][t][-1][2:4])
              else: # if id changes, means it is a new vdi vehicle in time t, prediction data of original vdi vehicle is missing
                vdi_prediction_dict[f'vdi_{i+1}'].append(np.array([0., 0.]))
          else: # if vdi_i is missing in current time step, then it has no prediction data
            vdi_prediction_dict[f'vdi_{i+1}'] = [np.array([0., 0.])] * self._predict_horizen
    
      # 2. trans data from global frame to ego frame
      ego_current_location, ego_current_heading = true_episode['ego'][time_step][-1][2:4], true_episode['ego'][time_step][-1][4]
      # ego state and its prediction data from global to ego
      ego_state_ego_frame = localize_vector_transform_list(ego_current_location, ego_current_heading, episode['ego'][time_step])
      ego_state_ego_frame_list.append(ego_state_ego_frame)
      if self._predict_task:
        ego_prediction_ego_frame, _ = localize_transform_list(ego_current_location, ego_current_heading, ego_prediction)
        ego_prediction_ego_frame_list.append(ego_prediction_ego_frame)
      # vdi state and prediction data from global frame to ego frame
      for i in range(self._vdi_num):
        if episode['mask_vdi'][time_step][i] == 1: # only trans vdi which is in the detection range
          vdi_state_ego_frame = localize_vector_transform_list(ego_current_location, ego_current_heading, episode[f'vdi_{i+1}'][time_step])
          vdi_state_ego_frame_dict[f'vdi_{i+1}'].append(vdi_state_ego_frame)
          if self._predict_task:
            # mask prediction data, TODO: waste of calculation
            prediction_mask = np.array([pred != [0., 0.] for pred in vdi_prediction_dict[f'vdi_{i+1}']])
            # print('vdi prediction mask:', prediction_mask)
            vdi_prediction_ego_frame, _ = localize_transform_list(ego_current_location, ego_current_heading, vdi_prediction_dict[f'vdi_{i+1}'])
            masked_vdi_prediction = vdi_prediction_ego_frame * self._expand(prediction_mask, len(vdi_prediction_ego_frame[0]))
            vdi_prediction_ego_frame_dict[f'vdi_{i+1}'].append(masked_vdi_prediction)
        else: # for zero padded data, keep it zero
          vdi_state_ego_frame_dict[f'vdi_{i+1}'].append(episode[f'vdi_{i+1}'][time_step])
          if self._predict_task:
            vdi_prediction_ego_frame_dict[f'vdi_{i+1}'].append(vdi_prediction_dict[f'vdi_{i+1}'])
      # vpi state to ego frame
      for i in range(self._vpi_num):
        if episode['mask_vpi'][time_step][i] == 1: # only trans vpi which is in the detection range
          vpi_state_ego_frame = localize_vector_transform_list(ego_current_location, ego_current_heading, episode[f'vpi_{i+1}'][time_step])
          vpi_state_ego_frame_dict[f'vpi_{i+1}'].append(vpi_state_ego_frame)
        else:
          vpi_state_ego_frame_dict[f'vpi_{i+1}'].append(episode[f'vpi_{i+1}'][time_step])
    
    # 3. update episode data which is used in training
    episode['ego'] = ego_state_ego_frame_list
    if self._predict_task:
      episode['ego_prediction'] = ego_prediction_ego_frame_list
    for i in range(self._vdi_num):
      episode[f'vdi_{i+1}'] = vdi_state_ego_frame_dict[f'vdi_{i+1}']
      if self._predict_task:
        episode[f'vdi_{i+1}_prediction'] = vdi_prediction_ego_frame_dict[f'vdi_{i+1}']
    for i in range(self._vpi_num):
      episode[f'vpi_{i+1}'] = vpi_state_ego_frame_dict[f'vpi_{i+1}']

    episode = {key: embodied.convert(value) for key, value in episode.items()} # convert data type

    # add it up
    self._total_steps += length
    self._loaded_steps += length
    self._total_episodes += 1
    self._loaded_episodes += 1

    # save replay file
    filename = save_episode(self._directory, episode)
    self._complete_eps[str(filename)] = episode
    self._enforce_limit()

  def save(self):
    pass
  
  def load(self, data_key=None):
    return load_episodes(self._directory, self._capacity, self._minlen)

  # def dataset(self): # batch=16, length=64 by default
  #   example = next(iter(self._generate_chunks(self._batch_length)))

  #   # print('example', type(example))
  #   dataset = tf.data.Dataset.from_generator(
  #       lambda: self._generate_chunks(self._batch_length),
  #       {k: v.dtype for k, v in example.items()},
  #       {k: v.shape for k, v in example.items()})
  #   # print('dataset1', dataset)
  #   dataset = dataset.batch(self._batch_size, drop_remainder=True)
  #   # print('dataset2', dataset)
  #   dataset = dataset.prefetch(5) # prepare for the next 5 batches, so that the GPU can be fully utilized
  #   # print('dataset3', dataset)
  #   return dataset

  def dataset(self): # self._generate_chunks
    sequence = self._sample_sequence()
    # print('sequence', sequence.keys())
    while True: # every 'next'
      chunk = collections.defaultdict(list)
      added = 0
      while added < self._batch_length:
        needed = self._batch_length - added
        adding = {k: v[:needed] for k, v in sequence.items()}
        sequence = {k: v[needed:] for k, v in sequence.items()}
        # print('adding', adding.keys())
        for key, value in adding.items():
          chunk[key].append(value)
        added += len(adding['action'])
        # print('added', added)

        # If the sequence is not enough, sample a new one.
        if len(sequence['action']) < 1:
          sequence = self._sample_sequence()
      
      # print('chunk1', chunk.keys())
      chunk = {k: np.concatenate(v) for k, v in chunk.items()}
      yield chunk

  def _sample_sequence(self):
    # get complete episodes
    episodes = list(self._complete_eps.values())
    if self._ongoing:
      episodes += [
          x for x in self._ongoing_eps.values()
          if eplen(x) >= self._minlen]
    episode = self._random.choice(episodes)
    total = len(episode['action'])
    length = total
    if self._maxlen:
      length = min(length, self._maxlen)
    # Randomize length to avoid all chunks ending at the same time in case the episodes are all of the same length.
    length -= np.random.randint(self._minlen)
    length = max(self._minlen, length)
    upper = total - length + 1
    if self._prioritize_ends:
      upper += self._minlen
    index = min(self._random.randint(upper), total - length)
    # embodied.convert like in org dmv3 chunk replay
    sequence = {
        k: embodied.convert(v[index: index + length])
        for k, v in episode.items() if not k.startswith('log_')}
    sequence['is_first'] = np.zeros(len(sequence['action']), np.bool_)
    sequence['is_first'][0] = True
    if self._maxlen:
      assert self._minlen <= len(sequence['action']) <= self._maxlen
    return sequence

  def _enforce_limit(self):
    # FIFO the redundant episodes
    if not self._capacity:
      return
    while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
      # Relying on Python preserving the insertion order of dicts.
      oldest, episode = next(iter(self._complete_eps.items()))
      self._loaded_steps -= eplen(episode)
      self._loaded_episodes -= 1
      del self._complete_eps[oldest]

  def _expand(self, value, dims):
    # print(value.shape, dims)
    while len(value.shape) < dims:
        value = value[..., None]
    # print(value)
    return value