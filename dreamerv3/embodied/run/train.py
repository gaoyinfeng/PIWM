import re

import embodied
import numpy as np


def train(agent, env, replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)

  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps) # every [batch_steps/train_ratio] steps train once. train_ratio = 32, batch_steps = size*length = 16*64 = 1024 by default
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every) # used for multiple GPU devices

  step = logger.step # a embodied.Counter()
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  def per_episode(ep):
    # record episode sumed reward
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    # record some statistic terms for i-sim driving scenes
    avg_speed = float(ep['sta_speed'].astype(np.float64).mean())
    collision_ticks = float(ep['sta_collision'].astype(np.float64).sum())
    completion = float(ep['sta_complet'][-1].astype(np.float64).mean())

    # sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())
    logger.add({
        'length': length,
        'score': score,
        'avg_speed': avg_speed,
        'collision_ticks': collision_ticks,
        'completion': completion,
        # 'sum_abs_reward': sum_abs_reward,
        # 'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
    }, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}. Complete {completion:.2f} of the route and average speed is {avg_speed:.1f} m/s. Collision ticks is {collision_ticks}.')

    # record episode stats
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(replay.add)

  print('Prefill train dataset.')
  random_agent = embodied.RandomAgent(env.act_space)
  # random fill replay max of a batch(1024) or train_fill(0 by default)
  while len(replay) < max(args.batch_steps, args.train_fill):
    driver(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset = agent.dataset(replay.dataset)

  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset'):
        batch[0] = next(dataset)
        # print('shape:', batch[0]['ego'].shape)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    # log the metrics
    if should_log(step):
      agg = metrics.result()
      # report = agent.report(batch[0])
      # report = {k: v for k, v in report.items() if 'train/' + k not in agg}
      logger.add(agg)
      # logger.add(report, prefix='report')
      logger.add(replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  timer.wrap('checkpoint', checkpoint, ['save', 'load'])
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  # save when every ep ends when using 'episode' type of replay
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  while step < args.steps:
    driver(policy, steps=100)
    # regular save
    if should_save(step):
      checkpoint.save()
    # save every 100k steps
    if not step.value % 100e3:
      filename = logdir / ('checkpoint_' + str(int(step.value / 100e3)) + '00k.ckpt')
      checkpoint.save(filename=filename)
  logger.write()
