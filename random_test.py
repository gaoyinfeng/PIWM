
import argparse
import numpy as np
import random
import time
from client_interface import ClientInterface

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='I-SIM Interface Test')
    parser.add_argument("map_name", type=str, default="DR_USA_Intersection_EP0", help="Name of the scenario (to identify map and folder for track files)", nargs='?')
    parser.add_argument("load_mode", type=str, default="vehicle", help="Dataset to load (vehicle, pedestrian, or both)", nargs='?')
    parser.add_argument("loader_type", type=str, default='prediction', help="prediction or dataset", nargs='?')
    parser.add_argument("state_frame", type=str, default="global", help="Vector state's frame, in ego frame or global frame", nargs='?')

    parser.add_argument("drive_as_record", type=bool, default=False, help="Ego is replayed as in record", nargs='?')
    parser.add_argument("continous_action", type=bool, default=False, help="Is the action type continous or discrete", nargs='?')
    parser.add_argument("control_steering", type=bool, default=False, help="Control both lon and lat motions", nargs='?')
    parser.add_argument("max_steps", type=int, default=None, help="max steps of one episode, None means orifinal max steps of the vehicle in xlm files", nargs='?')
    
    parser.add_argument("vdi_type", type=str, default='react', help="Default vdi type (react or record)", nargs='?')
    parser.add_argument("vdi_num", type=int, default=5, help="Default considered vdi num", nargs='?')
    parser.add_argument("vpi_num", type=int, default=5, help="Default considered far away vdi num", nargs='?')
    parser.add_argument("route_type", type=str, default='ground_truth', help="Default route type (predict, ground_truth or centerline)", nargs='?')

    parser.add_argument("visualization", type=bool, default=True, help="Visulize or not", nargs='?')
    parser.add_argument("ghost_visualization", type=bool, default=True, help="Render ghost(record) ego or not", nargs='?')
    parser.add_argument("route_visualization", type=bool, default=True, help="Render ego's route or not", nargs='?')
    parser.add_argument("route_bound_visualization", type=bool, default=False, help="Render ego's route bound or not", nargs='?')
    
    parser.add_argument("--port", type=int, default=8888, help="Number of the port (int)")
    parser.add_argument('--only_trouble', action="store_true", default=False, help='only select troubled vehicles in predictions as ego for testing')
    parser.add_argument('--eval', action="store_true", default=False, help='all possible ego vehicles are selected equal times')

    args = parser.parse_args()
    if args.map_name is None:
        raise IOError("You must specify a map. Type --help for help.")
    if args.load_mode != "vehicle" and args.load_mode != "pedestrian" and args.load_mode != "both":
        raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")
    
    # change env args
    args.drive_as_record = False
    args.state_frame = 'ego'
    args.loader_type = 'prediction'
    args.vdi_type = 'record'
    args.visualization = True
    args.route_visualization = True
    args.ghost_visualization = False
    args.eval = False
    args.vdi_num = 5
    args.vpi_num = 5

    # record overall result in list
    overall_score = []
    overall_avg_speed = []
    overall_collision_ticks = []
    overall_completion = []
    overall_rmse = []
    # episodes num and results
    total = 0
    success = 0
    collision = 0

    # initialize env
    env = ClientInterface(args, record=True)
    action_time_list = []
    env_step_time_list = []
    # start test
    while len(overall_score) < 156:
        state_dict = env.reset()
        # record infos of a episode in list
        step = 0
        total += 1
        ep_reward = [0.]
        ep_avg_speed = [0.]
        ep_collision_ticks = [0]
        ep_completion = [0]
        ep_rmse = [0]
        # vpi metrics
        surrounding_veh_num_list = []
        max_surrounding_veh_num = 0
        while True:
            action_dict = dict()
            for ego_id, ego_state in state_dict.items():
                action_time_1 = time.time()
                action = 3 # random.choice([1]) # random.randint(0,10) # action = policy(ego_state)
                action_time_2 = time.time()
                action_time_list.append(action_time_2 - action_time_1)
                action_dict[ego_id] = [action]
                #
                surrounding_veh_num = sum(ego_state['mask_vdi']) + sum(ego_state['mask_vpi'])
                surrounding_veh_num_list.append(surrounding_veh_num)
                if surrounding_veh_num > max_surrounding_veh_num:
                    max_surrounding_veh_num = surrounding_veh_num
            step_time_1 = time.time()
            next_state_dict, reward_dict, done_dict, aux_info_dict = env.step(action_dict, prediction=None)
            step_time_2 = time.time()
            env_step_time_list.append(step_time_2 - step_time_1)

            # only operate on one agent, so pick the first element
            ep_reward.append(list(reward_dict.values())[0])
            ep_collision_ticks.append(list(aux_info_dict.values())[0]['result'] == 'collision')
            ep_avg_speed.append(list(aux_info_dict.values())[0]['speed'])
            ep_completion.append(list(aux_info_dict.values())[0]['completion_rate'])
            ep_rmse.append(list(aux_info_dict.values())[0]['distance_to_gt'])

            # plus step
            step += 1
            state_dict = next_state_dict

            # check if the episode should end
            all_done = False not in done_dict.values()
            collision = list(aux_info_dict.values())[0]['result'] == 'collision'
            if all_done:
                # get metrics
                score = float(np.array(ep_reward).sum())
                avg_speed = float(np.array(ep_avg_speed).mean())
                collision_ticks = float(np.array(ep_collision_ticks).sum())
                completion = float(np.array(ep_completion[-1]).mean())
                rmse = float(np.sqrt(np.array(ep_rmse).astype(np.float64).mean()))

                # record in overall result list
                overall_score.append(score)
                overall_collision_ticks.append(collision_ticks)
                overall_completion.append(completion)
                overall_avg_speed.append(avg_speed)
                overall_rmse.append(rmse)
                collision_rate = np.mean([i> 0 for i in overall_collision_ticks])
                success_rate = np.mean([overall_completion[i] > 0.9 and overall_collision_ticks[i] < 1 for i in range(len(overall_completion))])
                
                # print episode result
                print(f'Episode {len(overall_completion)} has finished:')
                print(f'Episode has {len(ep_reward)} steps and return {score:.1f}. Complete {completion:.2f} of the route and average speed is {avg_speed:.1f} m/s. Collision ticks is {collision_ticks}.')
                # print(f'Ego ID is {state_dict.keys()}')
                # print(f'max surrounding vehcles num is {max_surrounding_veh_num}, average is {np.mean(surrounding_veh_num_list)}')
                # print(f'Collision is # {collision} #')
                print('_______________________' * 3)
                # print overall result
                print(f'Test Average: score is {np.mean(overall_score):.2f}, collision ticks is {np.mean(overall_collision_ticks):.2f}, compeltion is {np.mean(overall_completion):.4f}, average speed is {np.mean(overall_avg_speed):.2f}, avg rmse is {np.mean(overall_rmse):.2f}.')
                print(f'Overall result: collision_rate is {collision_rate:.4f}, success_rate is {success_rate:.4f}.')
                print(f'Average random policy time is {np.mean(action_time_list)}, average env step time is {np.mean(env_step_time_list)}')
                print('**********************' * 3)
                break
        


