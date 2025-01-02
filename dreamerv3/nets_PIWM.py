import re

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import jaxutils
from . import ninjax as nj
cast = jaxutils.cast_to_compute

''' Modified networks for Predictive Indivitial World Model (PIWM) '''

class PIWMRSSM(nj.Module):

  def __init__(
      self, shapes, pred_attention, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
      unimix=0.01, action_clip=1.0, **kw):
    
    # multi heads of rssm(ego, vdi_1...vdi_n, vpi_1....vpi_n)
    rssm_included = [k for k in shapes.keys() if k.startswith(('ego', 'vdi', 'vpi')) and 'prediction' not in k]
    self._keys = [k for k in shapes.keys() if k in rssm_included]
    
    # prediction attention network for merging vehicles' feature
    self._pred_attention = pred_attention # self-attention module, consider vehicles' interactive features when preducing z
    self._attention = self._pred_attention.out_dim # attention  dims

    # rssm network hyparameters
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial # 'learned' by config default, means zero-initial h and use learned nets produce z 
    self._unimix = unimix # true out = (1-unimix) * net out + unimix * random
    self._action_clip = action_clip
    self._kw = kw
  
  def initial(self, batch_size):
    state_dict = {}
    for key in self._keys:
      # initialize latent state(z) based on different branches if use learned weights to do it
      state = self._initial_single_state(key, batch_size)
      state_dict.update({key: state})
    return state_dict
  
  def _get_post(self, key, prior, pred_attention, embed):
    prefix = 'ego' if key.startswith('ego') else 'vdi' if key.startswith('vdi_') else 'vpi'
    x = jnp.concatenate([prior['deter'], pred_attention, embed], -1)
    x = self.get(f'{prefix}_obs_out', Linear, **self._kw)(x)
    stats = self._stats(f'{prefix}_obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    return {'stoch': stoch, 'deter': prior['deter'], **stats}

  def observe(self, embed, action, is_first, should_init_vdi, should_init_vpi, mask_vdi, mask_vpi, state_dict=None):
    # ego state and vdi state should be divided, also, vdi state should be divided into each vdi
    if state_dict is None: 
      state_dict = self.initial(batch_size=action.shape[0])

    # swap: change x shape from (16, 64, xx) to (64, 16, xx) (or from (64, 16, xx) to (16, 64, xx))
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    inputs = self._swap(action), self._swap(embed), self._swap(is_first), self._swap(should_init_vdi), self._swap(should_init_vpi), self._swap(mask_vdi), self._swap(mask_vpi)
    start = state_dict, state_dict

    post_dict, prior_dict = jaxutils.scan(step, inputs, start, self._unroll)

    # swap it back to the original shape (64, 16, xx)
    post_dict = {k: self._swap(v) for k, v in post_dict.items()}
    prior_dict = {k: self._swap(v) for k, v in prior_dict.items()}
    return post_dict, prior_dict

  def imagine(self, action, state_dict=None):
    state_dict = self.initial(action.shape[0]) if state_dict is None else state_dict
    assert isinstance(state_dict, dict), state_dict
    # swap: change x shape from (16, 64, xx) to (64, 16, xx), or from (64, 16, xx) to (16, 64, xx)
    action = self._swap(action)
    prior_dict, pred_attention_dict = jaxutils.scan(self.img_step, action, state_dict, self._unroll)
    # swap it back to the original shape (64, 16, xx)
    prior_dict = {k: self._swap(v) for k, v in prior_dict.items()}
    return prior_dict
  
  def obs_step(self, prev_state, prev_action, embed_dict, is_first, should_init_vdi, should_init_vpi, mask_vdi, mask_vpi):      
    # change data type
    is_first = cast(is_first)
    prev_action = cast(prev_action)
   
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
  
    # intialize latent state for first frame
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    # if is first frame, than state and action is set to 0, else keep it as it is

    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first), prev_state, self.initial(len(is_first)))
    # for state in first frame is then initialized by the rssm

    # initialize latent state for new or zero-padded(masked) vdis/vpis
    should_init_vdi, should_init_vpi = cast(should_init_vdi), cast(should_init_vpi)
    for key in prev_state.keys():
      if key.startswith('vdi_'):
        vdi_index = int(key.split('_')[-1])
        should_init = should_init_vdi[:, vdi_index - 1]
        # for vdis who are new to the observation, set its state to 0 first
        zero_paded_state = jax.tree_util.tree_map(lambda x: self._mask(x, 1.0 - should_init), prev_state[key])
        # and then initialize it using rssm initial function
        prev_state[key] = jax.tree_util.tree_map(lambda x, y: x + self._mask(y, should_init), zero_paded_state, self._initial_single_state(key, len(should_init)))
      elif key.startswith('vpi_'):
        vpi_index = int(key.split('_')[-1])
        should_init = should_init_vpi[:, vpi_index - 1]
        zero_paded_state = jax.tree_util.tree_map(lambda x: self._mask(x, 1.0 - should_init), prev_state[key])
        prev_state[key] = jax.tree_util.tree_map(lambda x, y: x + self._mask(y, should_init), zero_paded_state, self._initial_single_state(key, len(should_init)))

    # get prior ^z(t)
    prior_dict, pred_attention_dict = self.img_step(prev_state, prev_action, mask_vdi, mask_vpi)
    
    post_dict = {}# 
    # different branches
    for key in prior_dict.keys():
      if key.startswith(('ego', 'vdi_', 'vpi_')):
        prior = prior_dict[key]
        post = self._get_post(
            key,
            prior,
            pred_attention_dict[key],
            embed_dict[key]
        )
        post_dict.update({key: post})

    return cast(post_dict), cast(prior_dict) 
  
  def img_step(self, prev_state_dict, prev_action, mask_vdi, mask_vpi):
    # change data type
    prev_action = cast(prev_action)
    # action clip
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))

    def process_prior(self, key, feats_dict, pred_attention_dict, deter_dict):
      prefix = key.split('_')[0] if '_' in key else key
      x = jnp.concatenate([feats_dict[key], pred_attention_dict[key]], -1)
      x = self.get(f'{prefix}_img_out', Linear, **self._kw)(x)
      stats = self._stats(f'{prefix}_img_stats', x)
      dist = self.get_dist(stats)
      stoch = dist.sample(seed=nj.rng())
      
      return {
          'stoch': stoch,
          'deter': deter_dict[key]['deter'],
          **stats
      }

    # different branch for ego, vdi_1...vdi_n, vpi_1...vpi_n state
    deter_dict = {}
    for key in prev_state_dict.keys():
      if key.startswith(('ego', 'vdi_', 'vpi_')):
        prev_stoch = prev_state_dict[key]['stoch']
        deter = prev_state_dict[key]['deter']

        if self._classes:
          shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
          prev_stoch = prev_stoch.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch.shape):
          shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
          prev_action = prev_action.reshape(shape)

        prefix = key.split('_')[0] if '_' in key else key
        x = jnp.concatenate([prev_stoch, prev_action], -1)
        x = self.get(f'{prefix}_img_in', Linear, **self._kw)(x)
        x, deter = self._gru(f'{prefix}_gru', x, deter)
        deter_dict[key] = {'out': x, 'deter': deter}

    feats_dict = {k: v['out'] for k,v in deter_dict.items()}
    feats_dict.update({
        'mask_vdi': mask_vdi,
        'mask_vpi': mask_vpi
    })
    pred_attention_dict, pred_attention_dict_post, _ = self._pred_attention(feats_dict)

    prior_dict = {}
    for key in feats_dict.keys():
        if key.startswith(('ego', 'vdi_', 'vpi_')):
            prior = process_prior(self, key, feats_dict, pred_attention_dict, deter_dict)
            prior_dict[key] = prior

    return cast(prior_dict), cast(pred_attention_dict_post)

  # gets distribution from network outputs
  def get_dist(self, state, argmax=False):
    if self._classes:
      logit = state['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1)
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfp.MultivariateNormalDiag(mean, std)
  
  # gets initial distribution from network outputs
  def _get_stoch(self, key, deter, attention=None, weight='shared'):
    if weight == 'shared':
      x = jnp.concatenate([deter, attention], -1) if attention is not None else deter
      x = self.get('init_stoch_layer', Linear, **self._kw)(x)
      stats = self._stats('init_stoch_stats', x)
      dist = self.get_dist(stats)
    elif weight == 'branch':
      if key.startswith('ego'):
        x = jnp.concatenate([deter, attention], -1) if attention is not None else deter
        x = self.get('ego_img_out', Linear, **self._kw)(x)
        stats = self._stats('ego_img_stats', x)
        dist = self.get_dist(stats)
      elif key.startswith('vdi_'):
        x = jnp.concatenate([deter, attention], -1) if attention is not None else deter
        x = self.get('vdi_img_out', Linear, **self._kw)(x)
        stats = self._stats('vdi_img_stats', x)
        dist = self.get_dist(stats)
      elif key.startswith('vpi_'):
        x = jnp.concatenate([deter, attention], -1) if attention is not None else deter
        x = self.get('vpi_img_out', Linear, **self._kw)(x)
        stats = self._stats('vpi_img_stats', x)
        dist = self.get_dist(stats)
    return cast(dist.mode())
  
  def _initial_single_state(self, key, batch_size):
    # discrete or continuous latent space
    if self._classes:
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32),
          logit=jnp.zeros([batch_size, self._stoch, self._classes], f32),
          stoch=jnp.zeros([batch_size, self._stoch, self._classes], f32))
    else:
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32),
          mean=jnp.zeros([batch_size, self._stoch], f32),
          std=jnp.ones([batch_size, self._stoch], f32),
          stoch=jnp.zeros([batch_size, self._stoch], f32))
    # weights initialization
    # NOTE: we found that initialize latent states with 'zeros' or 'shared'(trick?)
    # weights may helps stablize early training, but after all these 3 methods preform almost equaly at last
    if self._initial == 'zeros': # remain zeros
      state = cast(state)
    elif self._initial == 'learned': # use learned network to initialize
      deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], batch_size, 0)
      attention = self.get('initial_attention', jnp.zeros, (self._attention,), f32) # 
      attention = jnp.repeat(jnp.tanh(attention)[None], batch_size, 0) # 
      state['stoch'] = self._get_stoch(key, cast(state['deter']), cast(attention), weight='branch')
      state = cast(state)
    else:
      raise NotImplementedError(self._initial)
    return state
  
  # gru cell
  def _gru(self, name, x, deter):
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = jnp.concatenate([deter, x], -1)
    x = self.get(name, Linear, **kw)(x)

    # GRU reset progress 
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)

    # GRU update progress
    deter = update * cand + (1 - update) * deter
    return deter, deter

  def _stats(self, name, x):
    if self._classes: 
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix: 
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

  def _swap(self, input):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if isinstance(input, dict):
      return {k: swap(v) for k, v in input.items()}
    else:
      return swap(input)
    
  def dyn_loss(self, post_dict, prior_dict, mask_vdi=1, mask_vpi=1, impl='kl', free=1.0):
    # get loss dims
    loss_dims = post_dict['ego']['deter'].shape[:2]
    loss_dict = {}
    # sum up losses for vdis or vpis branch
    vdi_dyn_kl_loss = jnp.zeros(loss_dims)
    vpi_dyn_kl_loss = jnp.zeros(loss_dims)
    for key in post_dict.keys():
      # how to calculate dyn loss, impl is 'kl' by default
      if impl == 'kl':
        loss = self.get_dist(sg(post_dict[key])).kl_divergence(self.get_dist(prior_dict[key]))
      elif impl == 'logprob':
        loss = -self.get_dist(prior_dict[key]).log_prob(sg(post_dict[key]['stoch']))
      else:
        raise NotImplementedError(impl)
      # kl free
      if free:
        loss = jnp.maximum(loss, free)
      # vdi sum up
      if key.startswith('vdi_'):
        vdi_index = int(key.split('_')[-1])
        mask = mask_vdi[:, :, vdi_index - 1]
        vdi_dyn_kl_loss += loss * mask
        loss_dict.update({'vdi_dyn_kl': vdi_dyn_kl_loss})
      # vpi sum up
      elif key.startswith('vpi_'):
        vpi_index = int(key.split('_')[-1])
        mask = mask_vpi[:, :, vpi_index - 1]
        vpi_dyn_kl_loss += loss * mask
        loss_dict.update({'vpi_dyn_kl': vpi_dyn_kl_loss})
      # ego
      else:
        loss_dict.update({key + '_dyn_kl': loss})
    # print(loss_dict.keys())
    return loss_dict

  def rep_loss(self, post_dict, prior_dict, mask_vdi=1, mask_vpi=1, impl='kl', free=1.0):
    # get loss dims
    loss_dims = post_dict['ego']['deter'].shape[:2]
    loss_dict = {}
    # sum up losses for vdis or vpis branch
    vdi_rep_kl_loss = jnp.zeros(loss_dims)
    vpi_rep_kl_loss = jnp.zeros(loss_dims)
    for key in post_dict.keys():
      # how to calculate rep loss, impl is 'kl' by default
      if impl == 'kl':
        loss = self.get_dist(post_dict[key]).kl_divergence(self.get_dist(sg(prior_dict[key])))
      elif impl == 'uniform':
        uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior_dict[key])
        loss = self.get_dist(post_dict[key]).kl_divergence(self.get_dist(uniform))
      elif impl == 'entropy':
        loss = -self.get_dist(post_dict[key]).entropy()
      elif impl == 'none':
        loss = jnp.zeros(post_dict[key]['deter'].shape[:-1])
      else:
        raise NotImplementedError(impl)
      # kl free
      if free:
        loss = jnp.maximum(loss, free)
      # vdi sum up
      if key.startswith('vdi_'):
        vdi_index = int(key.split('_')[-1])
        mask = mask_vdi[:, :, vdi_index - 1]
        vdi_rep_kl_loss += loss * mask
        loss_dict.update({'vdi_rep_kl': vdi_rep_kl_loss})
      # vpi sum up
      elif key.startswith('vpi_'):
        vpi_index = int(key.split('_')[-1])
        mask = mask_vpi[:, :, vpi_index - 1]
        vpi_rep_kl_loss += loss * mask
        loss_dict.update({'vpi_rep_kl': vpi_rep_kl_loss})
      # ego
      else:
        loss_dict.update({key + '_rep_kl': loss})
    # print(loss_dict.keys())
    return loss_dict


class PIWMEncoder(nj.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
      mlp_units=512, cnn='resize', cnn_depth=48,
      cnn_blocks=2, resize='stride',
      symlog_inputs=False, minres=4, **kw):
    
    excluded = ('is_first', 'is_last')

    shapes = {k: v for k, v in shapes.items() if (k not in excluded and not k.startswith('log_') and not k.endswith('_prediction'))}

    self.cnn_shapes = {k: v for k, v in shapes.items() if (
        len(v) == 3 and re.match(cnn_keys, k))}
    
    self.mlp_shapes = {k: v for k, v in shapes.items() if (
        len(v) in (1, 2) and re.match(mlp_keys, k))}

    self.shapes = {**self.cnn_shapes, **self.mlp_shapes} 
    print('Encoder CNN shapes:', self.cnn_shapes)
    print('Encoder MLP shapes:', self.mlp_shapes)
    
    cnn_kw = {**kw, 'minres': minres, 'name': 'cnn'}
    if self.cnn_shapes:
      if cnn == 'resnet':
        self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw)
      else:
        raise NotImplementedError(cnn)

    # mlp layers, 2 shared layers for trajectory and 2 layers for each branch
    if self.mlp_shapes:
      # vehicle info
      enc_mlp_layer = int(mlp_layers / 2)

      # encode trajectory using the same mlp
      self._traj_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=symlog_inputs, name='traj_mlp')

      # for ego, vdi and vpi features, using different mlp
      self._ego_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=False, name='ego_mlp')
      self._vdi_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=False, name='vdi_mlp')
      self._vpi_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=False, name='vpi_mlp')

  def __call__(self, data):
  
    # to get batch dims, and reshape the data 
    some_key, some_shape = list(self.shapes.items())[0]
    batch_dims = data[some_key].shape[:-len(some_shape)]
    data = {
        k: v.reshape((-1,) + v.shape[len(batch_dims):])
        for k, v in data.items()}

    outputs_dict = {}

    # for image inputs
    if self.cnn_shapes:
      inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1)
      output = self._cnn(inputs)
      output = output.reshape((output.shape[0], -1))
      outputs_dict.update({'cnn': output})

    # for vector inputs (vehicle and map)
    if self.mlp_shapes:
      for key in self.mlp_shapes:
        ob = jaxutils.cast_to_compute(data[key].astype(f32))
        if key.startswith(('ego', 'vdi_', 'vpi_')):
          traj_features = self._traj_mlp(ob)
          traj_features = traj_features.reshape((traj_features.shape[0], -1))
          if key.startswith('ego'):
            features = self._ego_mlp(traj_features)
          elif key.startswith('vdi_'):
            features = self._vdi_mlp(traj_features)
          elif key.startswith('vpi_'):
            features = self._vpi_mlp(traj_features)
        else:
          # features = self._map_mlp(ob)
          pass
        
        outputs_dict.update({key: features.reshape(batch_dims + features.shape[1:])})
    
    return outputs_dict


def multi_head_attention(q, k, v, mask, drop_out=0.1):
  d_k = q.shape[-1]
  att_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)
  att_logits = jnp.where(mask, att_logits, -1e9)
  attention = jax.nn.softmax(att_logits, axis=-1)
  if drop_out:
    # attention = dropout(attention)
    pass
  out = jnp.matmul(attention, v)

  return out, attention

class PredictAttention(nj.Module):
  # self-attention, produce ego and vdi's attention value for future prediciton task
  def __init__(
    self, shape, layers, heads, units_per_head, inputs=['tensor'], dims=None, 
    symlog_inputs=False, **kw):

    # data shape type transition
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)
    
    # network hyparameters
    self._shape = shape
    self._layers = layers
    self._heads = heads
    self._units_per_head = units_per_head
    self._inputs = Input(inputs, dims=dims)
    self._symlog_inputs = symlog_inputs
    # key words for dense layers and output(distribution) layers
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}
    self.out_dim = self._heads * self._units_per_head

  def __call__(self, inputs_dict):
    # preprocess inputs
    feature_dict = {key: value for key, value in inputs_dict.items() if key.startswith(('ego', 'vdi_', 'vpi_'))}
    if self._symlog_inputs:
      feature_dict = {key: jaxutils.symlog(value) for key, value in feature_dict.items()}
    # use attention mechanism to fuse ego and vdi features together
    vdi_mask = inputs_dict['mask_vdi']
    vdi_num = vdi_mask.shape[-1]
    vpi_mask = inputs_dict['mask_vpi']
    vpi_num = vpi_mask.shape[-1]
    # concat ego and vdi features together in entity dimension
    feature_dict = {key: jnp.expand_dims(value, axis=-2) for key, value in feature_dict.items()}
    vehicle_features_q = jnp.concatenate([feature_dict['ego']] + [feature_dict[f'vdi_{i+1}'] for i in range(vdi_num)] + [feature_dict[f'vpi_{i+1}'] for i in range(vpi_num)], axis=-2)
    vehicle_features_kv = jnp.concatenate([feature_dict['ego']] + [feature_dict[f'vdi_{i+1}'] for i in range(vdi_num)] + [feature_dict[f'vpi_{i+1}'] for i in range(vpi_num)], axis=-2)

    q_x = jaxutils.cast_to_compute(vehicle_features_q)
    kv_x = jaxutils.cast_to_compute(vehicle_features_kv)
    # attention
    # Dimensions: Batch*Length, entity, head, feature_per_head
    q = self.get('query', Linear, units=self._heads*self._units_per_head, **self._dense)(q_x).reshape([-1, 1 + vdi_num + vpi_num, self._heads, self._units_per_head])
    k = self.get('key', Linear, units=self._heads*self._units_per_head, **self._dense)(kv_x).reshape([-1, 1 + vdi_num + vpi_num, self._heads, self._units_per_head])
    v = self.get('value', Linear, units=self._heads*self._units_per_head, **self._dense)(kv_x).reshape([-1, 1 + vdi_num + vpi_num, self._heads, self._units_per_head])

    # Dimensions: Batch*Length, head, entity, feature_per_head
    q = q.transpose(0,2,1,3)
    k = k.transpose(0,2,1,3)
    v = v.transpose(0,2,1,3)

    # mask Dimensions: Batch*Length, head, 1, entity
    ego_mask = jnp.ones(list(q.shape[:1]) + [1,1])
    vdi_mask = vdi_mask.reshape(-1, 1, vdi_num)
    vpi_mask = vpi_mask.reshape(-1, 1, vpi_num)
    mask = jnp.concatenate([ego_mask, vdi_mask, vpi_mask], axis=-1).reshape([-1, 1, 1, 1 + vdi_num + vpi_num])
    mask = jnp.repeat(mask, self._heads, axis=1)

    # since they do different tasks in latter parts(vdi for future prediction only and ego for actor/critic/reward/count)
    # we use different mlp head to get a different attention result for ego and vdi for now
    self_attention_out, self_attention_mat = multi_head_attention(q, k, v, mask, drop_out=False)
    # Dimensions(back to): Batch*Length, entity, head, feature_per_head
    self_attention_out = self_attention_out.transpose(0,2,1,3)
    self_attention_mat = self_attention_mat.transpose(0,2,1,3)

    self_attention_out_dict = {}
    self_attention_mat_dict = {}
    for i in range(vdi_num + vpi_num + 1):
      if i == 0:
        # attention matrix for ego
        self_attention_mat_dict['ego'] = self_attention_mat[..., 0, :, :]
        # attention output for ego
        x = self_attention_out[..., 0, :, :]
        x = x.reshape([x.shape[0], -1])
        x = self.get('ego_out_mlp', Linear, units=self._heads*self._units_per_head, **self._dense)(x)
        self_attention_out_dict['ego'] = x.reshape(list(vehicle_features_q.shape[:-2]) + [-1])
      elif i < vdi_num + 1:
        # attention matrix for vdi
        self_attention_mat_dict[f'vdi_{i}'] = self_attention_mat[..., i, :, :]
        # attention output for vdi
        x = self_attention_out[..., i, :, :]
        x = x.reshape([x.shape[0], -1])
        x = self.get('vdi_out_mlp', Linear, units=self._heads*self._units_per_head, **self._dense)(x)
        self_attention_out_dict[f'vdi_{i}'] = x.reshape(list(vehicle_features_q.shape[:-2]) + [-1])
      else:
        # attention matrix for vpi
        self_attention_mat_dict[f'vpi_{i-vdi_num}'] = self_attention_mat[..., i, :, :]
        # attention output for vpi
        x = self_attention_out[..., i, :, :]
        x = x.reshape([x.shape[0], -1])
        x = self.get('vpi_out_mlp', Linear, units=self._heads*self._units_per_head, **self._dense)(x)
        self_attention_out_dict[f'vpi_{i-vdi_num}'] = x.reshape(list(vehicle_features_q.shape[:-2]) + [-1])
    return self_attention_out_dict, self_attention_out_dict, self_attention_mat_dict

class EgoAttention(nj.Module):
  # cross-attention or ego-attention, produce attention value for ego's task of actor/critic/predicting reward/predicting count
  def __init__(
    self, shape, layers, heads, units_per_head, inputs=['tensor'], dims=None, 
    symlog_inputs=False, **kw):

    # data shape type transition
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)
    # network hyparameters
    self._shape = shape
    self._layers = layers
    self._heads = heads
    self._units_per_head = units_per_head
    self._inputs = Input(inputs, dims=dims)
    self._symlog_inputs = symlog_inputs
    # key words for dense layers and output(distribution) layers
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, inputs_dict):
    # preprocess inputs
    feature_dict = {key: self._inputs(value) for key, value in inputs_dict.items() if isinstance(value, dict) and key != 'attention_vdi'}
    # vdi(and ego) feature contains its individual feature and its self-attention (prediction attention) to surrounding vehicles, as interactive features
    if self._symlog_inputs:
      feature_dict = {key: jaxutils.symlog(value) for key, value in feature_dict.items()}
    # use attention mechanism to fuse ego and vdi features together
    vdi_mask = inputs_dict['mask_vdi']
    vdi_num = vdi_mask.shape[-1]
    # concat ego and vdi features together in entity dimension
    feature_dict = {key: jnp.expand_dims(value, axis=-2) for key, value in feature_dict.items()}
    ego_features = feature_dict['ego']
    ego_features = jaxutils.cast_to_compute(ego_features)
    vehicle_features = jnp.concatenate([feature_dict['ego']] + [feature_dict[f'vdi_{i+1}'] for i in range(vdi_num)], axis=-2)
    vehicle_features = jaxutils.cast_to_compute(vehicle_features)
    # attention
    # Dimensions: Batch*Length, entity, head, feature_per_head
    q_ego = self.get('query', Linear, units=self._heads*self._units_per_head, **self._dense)(ego_features).reshape([-1, 1, self._heads, self._units_per_head])
    k_all = self.get('key', Linear, units=self._heads*self._units_per_head, **self._dense)(vehicle_features).reshape([-1, 1 + vdi_num, self._heads, self._units_per_head])
    v_all = self.get('value', Linear, units=self._heads*self._units_per_head, **self._dense)(vehicle_features).reshape([-1, 1 + vdi_num, self._heads, self._units_per_head])
    # Dimensions: Batch*Length, head, entity, feature_per_head
    q_ego = q_ego.transpose(0,2,1,3)
    k_all = k_all.transpose(0,2,1,3)
    v_all = v_all.transpose(0,2,1,3)
    # mask Dimensions: Batch*Length, head, 1, entity
    ego_mask = jnp.ones(list(q_ego.shape[:1]) + [1,1]) # Batch*Length, 1, 1
    vdi_mask = vdi_mask.reshape(-1, 1, vdi_num)
    mask = jnp.concatenate([ego_mask, vdi_mask], axis=-1).reshape([-1, 1, 1, vdi_num + 1])
    mask = jnp.repeat(mask, self._heads, axis=1)

    # since they do different tasks in latter parts(vdi for future prediction and ego for actor/critic/reward/count)
    # yet we use different mlp head to get a different attention result for ego and vdi for now
    ego_attention_out, ego_attention_mat = multi_head_attention(q_ego, k_all, v_all, mask, drop_out=False)
    # Dimensions(back to): Batch*Length, entity, head, feature_per_head
    ego_attention_out = ego_attention_out.transpose(0,2,1,3)
    ego_attention_mat = ego_attention_mat.transpose(0,2,1,3)

    # attention matrix for ego
    mat = ego_attention_mat[..., 0, :, :]
    # attention output for ego
    x = ego_attention_out[..., 0, :, :]
    x = x.reshape([x.shape[0], -1])
    x = self.get('ego_out_mlp', Linear, units=self._heads*self._units_per_head, **self._dense)(x)
    out = x.reshape(list(vehicle_features.shape[:-2]) + [-1]) # Batch, Length, out_feature

    return out, mat


class PredictionDecoder(nj.Module):

  def __init__(
      self, shapes, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*',
      mlp_layers=4, mlp_units=512, cnn='resize', cnn_depth=48, cnn_blocks=2,
      image_dist='mse', vector_dist='mse', resize='stride', bins=255,
      outscale=1.0, minres=4, cnn_sigmoid=False, **kw):
    
    # pick decode targets and their shapes
    excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
    shapes = {k: v for k, v in shapes.items() if k not in excluded}
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) in (1, 2) and k.endswith('_prediction')}
    
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Decoder CNN shapes:', self.cnn_shapes)
    print('Decoder MLP shapes:', self.mlp_shapes)

    # inputs preprocess
    self._inputs = Input(inputs, dims='deter')

    # decode image
    cnn_kw = {**kw, 'minres': minres, 'sigmoid': cnn_sigmoid}
    if self.cnn_shapes:
      shapes = list(self.cnn_shapes.values())
      assert all(x[:-1] == shapes[0][:-1] for x in shapes)
      shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
      if cnn == 'resnet':
        self._cnn = ImageDecoderResnet(
            shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name='cnn')
      else:
        raise NotImplementedError(cnn)
    self._image_dist = image_dist

    # decode vector
    mlp_kw = {**kw, 'dist': vector_dist, 'outscale': outscale, 'bins': bins}
    if self.mlp_shapes:
      # vehicle future trajectory prediction
      self._ego_mlp = MLP(self.mlp_shapes['ego_prediction'], mlp_layers, mlp_units, **mlp_kw, name='ego_mlp')
      self._vdi_mlp = MLP(self.mlp_shapes['vdi_1_prediction'], mlp_layers, mlp_units, **mlp_kw, name='vdi_mlp')

  def __call__(self, featrue_dict, drop_loss_indices=None):
    dists_dict = {}
    # decode image
    if self.cnn_shapes:
      feature = self._inputs(featrue_dict)
      if drop_loss_indices is not None:
        feature = feature[:, drop_loss_indices]
      flat = feature.reshape([-1, feature.shape[-1]])
      output = self._cnn(flat)
      output = output.reshape(feature.shape[:-1] + output.shape[1:])
      split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])
      means = jnp.split(output, split_indices, -1)
      dists_dict.update({
          key: self._make_image_dist(key, mean)
          for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
      
    # decode vector(future trajectory prediction)
    if self.mlp_shapes:
      for key in self.mlp_shapes:
        # input feature is vehicle feature concats its attention
        x = self._inputs(featrue_dict[key.replace('_prediction', '')])
        if key.startswith('ego'):
          dist = self._ego_mlp(x)
        elif key.startswith('vdi_'):
          dist = self._vdi_mlp(x)
        dists_dict.update({key: dist})
    return dists_dict

  def _make_image_dist(self, name, mean):
    mean = mean.astype(f32)
    if self._image_dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, 1), 3)
    if self._image_dist == 'mse':
      return jaxutils.MSEDist(mean, 3, 'sum')
    raise NotImplementedError(self._image_dist)

class PIWMMLP(nj.Module):

  # for modules like actor, critic, reward prediction, countinue prediction, we use ego's feature and ego-attention as input
  def __init__(
      self, shape, layers, units, inputs=['tensor'], dims=None, 
      symlog_inputs=False, **kw):

    # data shape type transition
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)

    # network hyparameters
    self._shape = shape   
    self._layers = layers 
    self._units = units   
    self._inputs = Input(inputs, dims=dims) 
    self._symlog_inputs = symlog_inputs     
    # key words for dense layers and output(distribution) layers
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, feature_dict, attention, concat_feat=False):
    feature_dict = {key: self._inputs(value) for key, value in feature_dict.items() if key.startswith(('ego', 'vdi_'))}
    print(feature_dict)
    if self._symlog_inputs:
      feature_dict = {key: jaxutils.symlog(value) for key, value in feature_dict.items() if key.startswith(('ego', 'vdi_'))}

    if concat_feat:
      feat = jnp.concatenate(list(feature_dict.values()), axis=-1)
    else: 
      if isinstance(attention, dict):
        attention = attention['ego']
      feat = jnp.concatenate([feature_dict['ego'], attention], axis=-1)
    x = jaxutils.cast_to_compute(feat)

    x = x.reshape([-1, x.shape[-1]])
    for i in range(self._layers):
      x = self.get(f'h{i}', Linear, self._units, **self._dense)(x)
    x = x.reshape(feat.shape[:-1] + (x.shape[-1],))

    if self._shape is None:
      return x
    elif isinstance(self._shape, tuple):
      return self._out('out', self._shape, x)
    elif isinstance(self._shape, dict):
      return {k: self._out(k, v, x) for k, v in self._shape.items()}
    else:
      raise ValueError(self._shape)

  def _out(self, name, shape, x):
    return self.get(f'dist_{name}', Dist, shape, **self._dist)(x)


class MLP(nj.Module):
  def __init__(
      self, shape, layers, units, inputs=['tensor'], dims=None,
      symlog_inputs=False, **kw):

    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)

    self._shape = shape
    self._layers = layers
    self._units = units
    self._inputs = Input(inputs, dims=dims)
    self._symlog_inputs = symlog_inputs

    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, inputs):
    feat = self._inputs(inputs)
    if self._symlog_inputs:
      feat = jaxutils.symlog(feat)
    x = jaxutils.cast_to_compute(feat)

    x = x.reshape([-1, x.shape[-1]])
    for i in range(self._layers):
      x = self.get(f'h{i}', Linear, self._units, **self._dense)(x)
    x = x.reshape(feat.shape[:-1] + (x.shape[-1],))

    if self._shape is None:
      return x
    elif isinstance(self._shape, tuple):
      return self._out('out', self._shape, x)
    elif isinstance(self._shape, dict):
      return {k: self._out(k, v, x) for k, v in self._shape.items()}
    else:
      raise ValueError(self._shape)

  def _out(self, name, shape, x):
    return self.get(f'dist_{name}', Dist, shape, **self._dist)(x)

class ImageEncoderResnet(nj.Module):

  def __init__(self, depth, blocks, resize, minres, **kw):
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._kw = kw

  def __call__(self, x):
    stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
    depth = self._depth
    x = jaxutils.cast_to_compute(x) - 0.5
    for i in range(stages):
      kw = {**self._kw, 'preact': False}
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, **kw)(x)
      elif self._resize == 'stride3':
        s = 2 if i else 3
        k = 5 if i else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, **kw)(x)
      elif self._resize == 'mean':
        N, H, W, D = x.shape
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
      elif self._resize == 'max':
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'same')
      else:
        raise NotImplementedError(self._resize)
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
      depth *= 2
    if self._blocks:
      x = get_act(self._kw['act'])(x)
    x = x.reshape((x.shape[0], -1))
    return x


class ImageDecoderResnet(nj.Module):

  def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
    self._shape = shape
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._sigmoid = sigmoid
    self._kw = kw

  def __call__(self, x):
    stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
    depth = self._depth * 2 ** (stages - 1)
    x = jaxutils.cast_to_compute(x)
    x = self.get('in', Linear, (self._minres, self._minres, depth))(x)
    for i in range(stages):
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
      depth //= 2
      kw = {**self._kw, 'preact': False}
      if i == stages - 1:
        kw = {}
        depth = self._shape[-1]
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, transp=True, **kw)(x)
      elif self._resize == 'stride3':
        s = 3 if i == stages - 1 else 2
        k = 5 if i == stages - 1 else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, transp=True, **kw)(x)
      elif self._resize == 'resize':
        x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
      else:
        raise NotImplementedError(self._resize)
    if max(x.shape[1:-1]) > max(self._shape[:-1]):
      padh = (x.shape[1] - self._shape[0]) / 2
      padw = (x.shape[2] - self._shape[1]) / 2
      x = x[:, int(np.ceil(padh)): -int(padh), :]
      x = x[:, :, int(np.ceil(padw)): -int(padw)]
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    if self._sigmoid:
      x = jax.nn.sigmoid(x)
    else:
      x = x + 0.5
    return x

'''Basic network units'''

def get_act(name):
  if callable(name):
    return name
  elif name == 'none':
    return lambda x: x
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)


class Input:

  def __init__(self, keys=['tensor'], dims=None):
    assert isinstance(keys, (list, tuple)), keys
    self._keys = tuple(keys)
    self._dims = dims or self._keys[0] # if dims has content then is dims, else is keys[0](most likely 'deter')

  def __call__(self, inputs):
    # make inputs as a dict
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    inputs = inputs.copy()

    # if inputs need softmax
    for key in self._keys:
      if key.startswith('softmax_'):
        inputs[key] = jax.nn.softmax(inputs[key[len('softmax_'):]])
    if not all(k in inputs for k in self._keys):
      needs = f'{{{", ".join(self._keys)}}}'
      found = f'{{{", ".join(inputs.keys())}}}'
      raise KeyError(f'Cannot find keys {needs} among inputs {found}.')
    
    # keep value in the same shape
    values = [inputs[k] for k in self._keys]
    dims = len(inputs[self._dims].shape)
    for i, value in enumerate(values):
      if len(value.shape) > dims:
        values[i] = value.reshape(
            value.shape[:dims - 1] + (np.prod(value.shape[dims - 1:]),))
        
    # recover data type since value data type may change in last step
    values = [x.astype(inputs[self._dims].dtype) for x in values]
    # concat input values (for example concat h and z as the input of decoder, reward and count heads)
    return jnp.concatenate(values, -1)


class Dist(nj.Module):

  def __init__(
      self, shape, dist='mse', outscale=0.1, outnorm=False, minstd=1.0,
      maxstd=1.0, unimix=0.0, bins=255):
    assert all(isinstance(dim, int) for dim in shape), shape
    self._shape = shape
    self._dist = dist
    self._minstd = minstd
    self._maxstd = maxstd
    self._unimix = unimix
    self._outscale = outscale
    self._outnorm = outnorm
    self._bins = bins

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    kw = {}
    kw['outscale'] = self._outscale
    kw['outnorm'] = self._outnorm
    shape = self._shape
    # discrete output value shape for critic & reward head
    if self._dist.endswith('_disc'):
      shape = (*self._shape, self._bins)

    # mlp -> mean
    out = self.get('out', Linear, int(np.prod(shape)), **kw)(inputs)
    out = out.reshape(inputs.shape[:-1] + shape).astype(f32)

    # mlp -> std
    if self._dist in ('normal', 'trunc_normal'):
      std = self.get('std', Linear, int(np.prod(self._shape)), **kw)(inputs)
      std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)

    # return distribution
    if self._dist == 'symlog_mse': # decoder
      return jaxutils.SymlogDist(out, len(self._shape), 'mse', 'sum')
    if self._dist == 'symlog_disc': # critic & reward
      return jaxutils.DiscDist(
          out, len(self._shape), -20, 20, jaxutils.symlog, jaxutils.symexp)
    if self._dist == 'mse': 
      return jaxutils.MSEDist(out, len(self._shape), 'sum')
    if self._dist == 'normal':
      lo, hi = self._minstd, self._maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.Normal(jnp.tanh(out), std)
      dist = tfd.Independent(dist, len(self._shape))
      dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
      return dist
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'onehot':
      if self._unimix:
        probs = jax.nn.softmax(out, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        out = jnp.log(probs)
      dist = jaxutils.OneHotDist(out)
      if len(self._shape) > 1:
        dist = tfd.Independent(dist, len(self._shape) - 1)
      dist.minent = 0.0
      dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
      return dist
    raise NotImplementedError(self._dist)


class Conv2D(nj.Module):

  def __init__(
      self, depth, kernel, stride=1, transp=False, act='none', norm='none',
      pad='same', bias=True, preact=False, winit='uniform', fan='avg'):
    self._depth = depth
    self._kernel = kernel
    self._stride = stride
    self._transp = transp
    self._act = get_act(act)
    self._norm = Norm(norm, name='norm')
    self._pad = pad.upper()
    self._bias = bias and (preact or norm == 'none')
    self._preact = preact
    self._winit = winit
    self._fan = fan

  def __call__(self, hidden):
    if self._preact:
      hidden = self._norm(hidden)
      hidden = self._act(hidden)
      hidden = self._layer(hidden)
    else:
      hidden = self._layer(hidden)
      hidden = self._norm(hidden)
      hidden = self._act(hidden)
    return hidden

  def _layer(self, x):
    if self._transp:
      shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
      kernel = self.get('kernel', Initializer(
          self._winit, fan=self._fan), shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_transpose(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWOI', 'NHWC'))
    else:
      shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
      kernel = self.get('kernel', Initializer(
          self._winit, fan=self._fan), shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_general_dilated(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if self._bias:
      bias = self.get('bias', jnp.zeros, self._depth, np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    return x


class Linear(nj.Module):

  def __init__(
      self, units, act='none', norm='none', bias=True, outscale=1.0,
      outnorm=False, winit='uniform', fan='avg'):
    self._units = tuple(units) if hasattr(units, '__len__') else (units,)
    self._act = get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'
    self._outscale = outscale
    self._outnorm = outnorm
    self._winit = winit
    self._fan = fan

  def __call__(self, x):
    shape = (x.shape[-1], np.prod(self._units))
    kernel = self.get('kernel', Initializer(
        self._winit, self._outscale, fan=self._fan), shape)
    kernel = jaxutils.cast_to_compute(kernel)
    x = x @ kernel
    if self._bias:
      bias = self.get('bias', jnp.zeros, np.prod(self._units), np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    if len(self._units) > 1:
      x = x.reshape(x.shape[:-1] + self._units)
    x = self.get('norm', Norm, self._norm)(x)
    x = self._act(x)
    return x

class Norm(nj.Module):

  def __init__(self, impl):
    self._impl = impl

  def __call__(self, x):
    dtype = x.dtype
    if self._impl == 'none':
      return x
    elif self._impl == 'layer':
      x = x.astype(f32)
      x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
      x *= self.get('scale', jnp.ones, x.shape[-1], f32)
      x += self.get('bias', jnp.zeros, x.shape[-1], f32)
      return x.astype(dtype)
    else:
      raise NotImplementedError(self._impl)


class Initializer:

  def __init__(self, dist='uniform', scale=1.0, fan='avg'):
    self.dist = dist
    self.scale = scale
    self.fan = fan

  def __call__(self, shape):
    if self.scale == 0.0:
      value = jnp.zeros(shape, f32)
    elif self.dist == 'uniform':
      fanin, fanout = self._fans(shape)
      denoms = {'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout}
      scale = self.scale / denoms[self.fan]
      limit = np.sqrt(3 * scale)
      value = jax.random.uniform(
          nj.rng(), shape, f32, -limit, limit)
    elif self.dist == 'normal':
      fanin, fanout = self._fans(shape)
      denoms = {'avg': np.mean((fanin, fanout)), 'in': fanin, 'out': fanout}
      scale = self.scale / denoms[self.fan]
      std = np.sqrt(scale) / 0.87962566103423978
      value = std * jax.random.truncated_normal(
          nj.rng(), -2, 2, shape, f32)
    elif self.dist == 'ortho':
      nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
      matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
      mat = jax.random.normal(nj.rng(), matshape, f32)
      qmat, rmat = jnp.linalg.qr(mat)
      qmat *= jnp.sign(jnp.diag(rmat))
      qmat = qmat.T if nrows < ncols else qmat
      qmat = qmat.reshape(nrows, *shape[:-1])
      value = self.scale * jnp.moveaxis(qmat, 0, -1)
    else:
      raise NotImplementedError(self.dist)
    return value

  def _fans(self, shape):
    if len(shape) == 0:
      return 1, 1
    elif len(shape) == 1:
      return shape[0], shape[0]
    elif len(shape) == 2:
      return shape
    else:
      space = int(np.prod(shape[:-2]))
      return shape[-2] * space, shape[-1] * space