import re

import embodied
import numpy as np


def eval_only(agent, env, logger, args, task):
  print('eval agent')

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])
  timer.wrap('logger', logger, ['write'])

  report_episode = 40
  score_list = []
  collision_ticks_list = []
  completion_list = []
  avg_speed_list = []
  rmse_list = []

  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    # record some statistic terms for i-sim driving scenes
    avg_speed = float(ep['sta_speed'].astype(np.float64).mean())
    collision_ticks = float(ep['sta_collision'].astype(np.float64).sum())
    completion = float(ep['sta_complet'][-1].astype(np.float64).mean())
    rmse = float(np.sqrt(ep['sta_gt_distance'].astype(np.float64).mean()))
    
    logger.add({
        'length': length,
        'score': score,
        'avg_speed': avg_speed,
        'collision_ticks': collision_ticks,
        'completion': completion,
        # 'sum_abs_reward': sum_abs_reward,
        # 'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
    }, prefix='episode')

    score_list.append(score)
    collision_ticks_list.append(collision_ticks)
    completion_list.append(completion)
    avg_speed_list.append(avg_speed)
    rmse_list.append(rmse)
    collision_rate = np.mean([i> 0 for i in collision_ticks_list])
    success_rate = np.mean([completion_list[i] > 0.9 and collision_ticks_list[i] < 1 for i in range(len(completion_list))])
    
    print(f'Episode {len(completion_list)} has finished:')
    print(f'Test Average: score is {np.mean(score_list):.2f}, collision ticks is {np.mean(collision_ticks_list):.2f}, compeltion is {np.mean(completion_list):.4f}, average speed is {np.mean(avg_speed_list):.2f}, rmse is {np.mean(rmse_list):.2f}.')
    print(f'Overall result: collision_rate is {collision_rate:.4f}, success_rate is {success_rate:.4f}.')
    print('**********************' * 3)

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

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  # load trained model weights
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation loop.')
  policy = lambda *args: agent.policy(*args, mode='eval')
  if 'prediction' in task:
    predictor = lambda *args: agent.predictor(*args)
  else:
    predictor = None
    
  while step < args.steps:
    driver(policy, episodes=report_episode, predictor=predictor)
    if should_log(step):
      logger.add(metrics.result())
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  logger.write()
