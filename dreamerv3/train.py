import importlib
import pathlib
import sys
import warnings
from functools import partial as bind
import yaml
import os

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from embodied import wrappers


def main(argv=None):
  from . import agent as agt

  current_dir = pathlib.Path(__file__).resolve().parent
  config_path = os.path.join(current_dir, 'configs.yaml')

  with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

  vdi_num   = config['interaction_prediction']['env']['interaction']['vdi_num']
  vpi_num   = config['interaction_prediction']['env']['interaction']['vpi_num']

  parsed, vpi = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(vpi)
  # args contains items in 'run', logdir, and batch size info
  args = embodied.Config(
      **config.run, logdir=config.logdir, 
      batch_steps=config.batch_size * config.batch_length)

  encoder_mlp_keys = ['ego'] + \
      [f'vdi_{i}' for i in range(1, vdi_num + 1)] + \
      [f'vpi_{i}' for i in range(1, vpi_num + 1)]
  decoder_mlp_keys = ['ego_prediction'] + \
      [f'vdi_{i}_prediction' for i in range(1, vdi_num + 1)]
  
  config = config.update({
    'encoder': {'mlp_keys': encoder_mlp_keys},
    'decoder': {'mlp_keys': decoder_mlp_keys},
  })

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  config.save(logdir / 'config.yaml')
  step = embodied.Counter()
  logger = make_logger(parsed, logdir, step, config)

  cleanup = []
  try:
    if args.script == 'train' and config.task.startswith('interaction'):
      # change replay from v3 version (save as chunks) to v2 version (save as episodes, which can be used to make prediction data)
      if args.replay_style == 'chunk':
        replay = make_replay(config, logdir / 'replay')
      elif args.replay_style == 'ep':
        replay = make_replay_ep(config, logdir / 'replay')
      env = make_envs(config) 
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train(agent, env, replay, logger, args)

    elif args.script == 'eval_only' and config.task.startswith('interaction'):
      # env is set to eval mode, every vehilce has the same oppotunity to be selected
      config = config.update({'env':{'interaction':{'eval': True}}})

      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.eval_only(agent, env, logger, args, config.task)

    elif args.script == 'train':
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train(agent, env, replay, logger, args)

    elif args.script == 'train_save':
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_save(agent, env, replay, logger, args)

    elif args.script == 'train_eval':
      replay = make_replay(config, logdir / 'replay')
      eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      eval_env = make_envs(config)
      cleanup += [env, eval_env]
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args)

    elif args.script == 'train_holdout':
      replay = make_replay(config, logdir / 'replay')
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
        eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_holdout(
          agent, env, replay, eval_replay, logger, args)

    elif args.script == 'eval_only':
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.eval_only(agent, env, logger, args)

    elif args.script == 'parallel':
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      step = embodied.Counter()
      env = make_env(config)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      env.close()
      replay = make_replay(config, logdir / 'replay', rate_limit=True)
      embodied.run.parallel(
          agent, replay, logger, bind(make_env, config),
          num_envs=config.envs.amount, args=args)

    else:
      raise NotImplementedError(args.script)
  finally:
    for obj in cleanup:
      obj.close()


def make_logger(parsed, logdir, step, config):
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  
  log_list = [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(logdir),
  ]
  if config.task.startswith('interaction'):
    log_list += [
      embodied.logger.JSONLOutput(logdir, 'avg_speed.jsonl', 'episode/avg_speed'),
      embodied.logger.JSONLOutput(logdir, 'completion.jsonl', 'episode/completion'),
      embodied.logger.JSONLOutput(logdir, 'collision_ticks.jsonl', 'episode/collision_ticks'),
    ]
  logger = embodied.Logger(step, log_list, multiplier)
  return logger


def make_replay(
    config, directory=None, is_eval=False, rate_limit=False, **kwargs):
  assert config.replay == 'uniform' or not rate_limit
  # set replay size
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  if config.replay == 'uniform' or is_eval:
    kw = {'online': config.replay_online}
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * config.batch_size
      kw['min_size'] = config.batch_size
    replay = embodied.replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay


def make_replay_ep(
    config, directory=None, is_eval=False, rate_limit=False):
  
  assert config.replay == 'uniform' or not rate_limit
  # set replay size
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  
  vdi_num = config.env.interaction.vdi_num
  vpi_num = config.env.interaction.vpi_num
  predict_horizen = config.env.interaction.predict_horizen # 20 means 2s in 10hz for prediction task

  replay = embodied.replay.ReplayEp(directory, capacity=size, batch_size=config.batch_size, batch_length=length, vdi_num=vdi_num, vpi_num=vpi_num, predict_horizen=predict_horizen,
                                    ongoing=False, minlen=config.batch_length, maxlen=config.batch_length, prioritize_ends=True)

  return replay


def make_envs(config, **overrides):
  print("Overrides:", overrides)
  suite, task = config.task.split('_', 1)
  ctors = []
  for index in range(config.envs.amount):
    ctor = lambda: make_env(config, **overrides) 
    if config.envs.parallel != 'none':
      ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
    if config.envs.restart:
      ctor = bind(wrappers.RestartOnException, ctor)
    ctors.append(ctor)
  envs = [ctor() for ctor in ctors]
  return embodied.BatchEnv(suite, envs, parallel=(config.envs.parallel != 'none'))


def make_env(config, **overrides):
  # You can add custom environments by creating and returning the environment
  # instance here. Environments with different interfaces can be converted
  # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
  suite, task = config.task.split('_', 1)
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'interaction': 'embodied.envs.interaction:Interaction',
  }[suite]
  
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  env = ctor(task, kwargs) if suite == 'interaction' else ctor(task, **kwargs)
  return wrap_env(env, config) 

def wrap_env(env, config):
  args = config.wrapper
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif space.discrete:
      env = wrappers.OneHotAction(env, name)
    elif args.discretize:
      env = wrappers.DiscretizeAction(env, name, args.discretize)
    else:
      env = wrappers.NormalizeAction(env, name)
  env = wrappers.ExpandScalars(env)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env


if __name__ == '__main__':
  main()
