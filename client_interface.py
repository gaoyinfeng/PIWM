import sys
import pathlib
import os
import zmq
import numpy as np
import random
import gym
import time
from datetime import datetime
import pickle
import collections

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

class Dict2Class(object):

    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])

class ClientInterface(object):

    def __init__(self, args, record=True):
        # env args
        self.args = Dict2Class(args) if isinstance(args, dict) else args
        self.discrete_action_num = 4
        self.action_space = gym.spaces.Discrete(self.discrete_action_num)

        # connection with I-SIM server
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        print("connecting to interaction gym...")
        url = ':'.join(["tcp://localhost", str(self.args.port)])
        self._socket.connect(url)
        
        # simulator statue flags
        self.env_init_flag = False
        self.can_change_track_file_flag = False
        self.scen_init_flag = False
        self.env_reset_flag = False
        self._gt_csv_index = None
        self.ego_id_list = None

        time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # record episode data if needed to calculate metrics
        self._record = record
        self.run_data_dict = dict()
        self._run_filename = "run " + time_string + '.pkl'
        # record prediction data if possible to calculate ade
        self.prediction_data_dict = dict()
        self._prediction_filename = "prediction " + time_string + '.pkl'

        save_dir = 'pkl_data'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._run_filename = os.path.join(save_dir, "run " + time_string + '.pkl')
        self._prediction_filename = os.path.join(save_dir, "prediction " + time_string + '.pkl')

    def __del__(self):
        self._socket.close()

    def _env_initialize(self, args):
        # settings in dict format
        settings = vars(args)
        # send to env
        message_send = {'command': 'env_init', 'content': settings}
        # print("in _env_initialize def(), settings: ",settings)
        self._socket.send_string(str(message_send))
        # recieve from env
        message_recv = self._socket.recv_string()

        if message_recv == 'env_init_done':
            self.env_init_flag = True
            print('env init done')

    def _change_track_file(self):
        # send to env
        # for simple react scenarios, the track file is fixed
        message_send = {'command': 'track_init'}
        self._socket.send_string(str(message_send))
        # recieve from env
        message_recv = self._socket.recv()
        str_message = bytes.decode(message_recv)
        self._gt_csv_index = eval(str_message)
        self.can_change_track_file_flag = False

    def _scenario_initialize(self):
        # send to env
        message_send = {'command': 'scen_init'}
        self._socket.send_string(str(message_send))

        # recieve from env
        message_recv = self._socket.recv()
        str_message = bytes.decode(message_recv)

        if str_message == 'wrong_num':
            print(message_recv)
        else:
            self.ego_id_list = eval(str_message)
            self.scen_init_flag = True
    
    def _reset_prepare(self):
        # init env
        if not self.env_init_flag:
            self._env_initialize(self.args)
        else:
            # change track file number
            if self.can_change_track_file_flag and not self.scen_init_flag:
                self._change_track_file()
            # choose ego vehicle and map init
            elif not self.scen_init_flag:
                self._scenario_initialize()
            # several depandecies have been checked, can reset environment
            elif self.scen_init_flag and not self.env_reset_flag:
                return True

        return False

    def _observation_to_ego_dict_array(self, observation, ego_id_list, vdi_id_dict, vpi_id_dict):
        if self.args.state_frame == 'ego':
            needed_state = ['vector_past_vehicle_state_ego_frame'] # ['vector_map']
        elif self.args.state_frame == 'global':
            needed_state = ['vector_past_vehicle_state'] # ['vector_map']

        # initialize ego state dict
        ego_state_dict = dict()
        for ego_id in ego_id_list:
            ego_state_dict[ego_id] = dict()
        # for every ego vehicle(since there may be multiple ego vehicles)
        for ego_id in ego_id_list:
            # vdi_1 ~ vdi_n, and mask_vdi indicates their existence
            vdi_index_set = set('vdi_' + str(i) for i in range(1, self.args.vdi_num + 1))
            mask_vdi = np.zeros(len(vdi_index_set), dtype=np.int32)
            is_new_vdi = np.zeros(len(vdi_index_set), dtype=np.int32)
            id_vdi = np.zeros(len(vdi_index_set), dtype=np.int32)
            # vpi_1 ~ vpi_n, also has mask
            vpi_index_set = set('vpi_' + str(i) for i in range(1, self.args.vpi_num + 1))
            mask_vpi = np.zeros(len(vpi_index_set), dtype=np.int32)
            is_new_vpi = np.zeros(len(vpi_index_set), dtype=np.int32)
            id_vpi = np.zeros(len(vpi_index_set), dtype=np.int32)

            # current surrounding vehicles' id, from closet to farest (distance), which is devided into 2 parts
            surrounding_vehicles_id = observation['surrounding_vehicles_id'][ego_id]
            # print(surrounding_vehicles_id)
            vdi_id_distance = list(surrounding_vehicles_id)[:self.args.vdi_num]
            vpi_id_distance = list(surrounding_vehicles_id)[self.args.vdi_num:self.args.vdi_num + self.args.vpi_num]
            # print(vdi_id_distance, vpi_id_distance)
                
            # get state for every ego vehicle
            for state_name in needed_state:
                state = observation[state_name]
                # vehicle state
                if state_name.startswith('vector_past_vehicle_state'):
                    # 1. remove or transfer(actually is also remove) vehicles' id from vdi_id_dict(id: vdi_*) and vpi_id_dict(id: vpi_*)
                    remove_vdi_id_set = set(vdi_id_dict.keys()) - set(vdi_id_distance)
                    for remove_vdi_id in remove_vdi_id_set:
                        vdi_id_dict.pop(remove_vdi_id)
                    remove_vpi_id_set = set(vpi_id_dict.keys()) - set(vpi_id_distance)
                    for remove_vpi_id in remove_vpi_id_set:
                        vpi_id_dict.pop(remove_vpi_id)

                    # 2. update ego_state_dict
                    for vehicle_id, traj_state in state.items():
                        vector_state = np.reshape(traj_state, [19, 5])
                        # ego state
                        if vehicle_id == ego_id: 
                            ego_state_dict[ego_id]['ego'] = vector_state
                        # vdi and vpi vehicle state
                        else: 
                            # if this vehicle has been in the state dict in last step and in ego's dectect range
                            if vehicle_id in vdi_id_dict.keys():
                                vdi_index = vdi_id_dict[vehicle_id]
                                ego_state_dict[ego_id][vdi_index] = vector_state
                            elif vehicle_id in vpi_id_dict.keys():
                                vpi_index = vpi_id_dict[vehicle_id]
                                ego_state_dict[ego_id][vpi_index] = vector_state
                            # if this vehicle is new for state dict, then put it in the state dict if there is a room
                            else:
                                if len(vdi_id_dict.keys()) < self.args.vdi_num and vehicle_id in vdi_id_distance:
                                    feasible_vdi_index = random.choice(list(vdi_index_set - set(vdi_id_dict.values())))
                                    is_new_vdi[int(feasible_vdi_index[-1]) - 1] = 1
                                    vdi_id_dict[vehicle_id] = feasible_vdi_index
                                    ego_state_dict[ego_id][feasible_vdi_index] = vector_state
                                elif len(vpi_id_dict.keys()) < self.args.vpi_num and vehicle_id in vpi_id_distance:
                                    feasible_vpi_index = random.choice(list(vpi_index_set - set(vpi_id_dict.values())))
                                    is_new_vpi[int(feasible_vpi_index[-1]) - 1] = 1
                                    vpi_id_dict[vehicle_id] = feasible_vpi_index
                                    ego_state_dict[ego_id][feasible_vpi_index] = vector_state

                    # 3. update vdi_mask and vpi_mask
                    for vehicle_id, vdi_index in vdi_id_dict.items():
                        mask_vdi[int(vdi_index[-1]) - 1] = 1
                        id_vdi[int(vdi_index[-1]) - 1] = vehicle_id
                    for vehicle_id, vpi_index in vpi_id_dict.items():
                        mask_vpi[int(vpi_index[-1]) - 1] = 1
                        id_vpi[int(vpi_index[-1]) - 1] = vehicle_id

                    # 4. complete state dict if there are not enough vehicles around
                    while len(ego_state_dict[ego_id].keys()) < self.args.vdi_num + self.args.vpi_num + 1:
                        current_vdi_index_set = set([key for key in ego_state_dict[ego_id].keys() if 'vdi' in key])
                        current_vpi_index_set = set([key for key in ego_state_dict[ego_id].keys() if 'vpi' in key])
                        if len(current_vdi_index_set) < self.args.vdi_num:
                            padding_vdi_index = list(vdi_index_set - current_vdi_index_set)[0]
                            ego_state_dict[ego_id][padding_vdi_index] = np.zeros([19, 5])
                        elif len(current_vpi_index_set) < self.args.vpi_num:
                            padding_vpi_index = list(vpi_index_set - current_vpi_index_set)[0]
                            ego_state_dict[ego_id][padding_vpi_index] = np.zeros([19, 5])

                    # 5. plus mask and vpi vdi padding related state in state dict
                    ego_state_dict[ego_id]['mask_vdi'] = mask_vdi
                    ego_state_dict[ego_id]['id_vdi'] = id_vdi
                    ego_state_dict[ego_id]['mask_vpi'] = mask_vpi
                    ego_state_dict[ego_id]['id_vpi'] = id_vpi

                    ego_state_dict[ego_id]['should_init_vdi'] = []
                    for i in range(len(mask_vdi)):
                        if not mask_vdi[i] or is_new_vdi[i]:
                            ego_state_dict[ego_id]['should_init_vdi'].append(1)
                        else:
                            ego_state_dict[ego_id]['should_init_vdi'].append(0)

                    ego_state_dict[ego_id]['should_init_vpi'] = []
                    for i in range(len(mask_vpi)):
                        if not mask_vpi[i] or is_new_vpi[i]:
                            ego_state_dict[ego_id]['should_init_vpi'].append(1)
                        else:
                            ego_state_dict[ego_id]['should_init_vpi'].append(0)

                    # 6. from closet to farest (distance) of vdi/vpi vehicles index
                    ego_state_dict[ego_id]['index_vdi'] = []
                    for vehicle_id in vdi_id_distance:
                        ego_state_dict[ego_id]['index_vdi'].append(vdi_id_dict[vehicle_id])
                    ego_state_dict[ego_id]['index_vpi'] = []
                    for vehicle_id in vpi_id_distance:
                        ego_state_dict[ego_id]['index_vpi'].append(vpi_id_dict[vehicle_id])

                elif state_name == 'vector_map':
                    ego_state_dict[ego_id]['map'] = np.reshape(state, [-1, 4])
                
        return ego_state_dict, vdi_id_dict, vpi_id_dict
    
    def reset(self):
        # reset flags
        self.can_change_track_file_flag = True  # this is used for multi-track-file random selection
        self.scen_init_flag = False
        self.env_reset_flag = False

        while not self._reset_prepare():
            self._reset_prepare()

        # send to env
        message_send = {'command': 'reset'}
        self._socket.send_string(str(message_send))
        # recieve from env
        message_recv = self._socket.recv()
        message_recv = eval(bytes.decode(message_recv))
        if isinstance(message_recv, dict):
            self.env_reset_flag = True
            observation = message_recv['observation']
        else:
            self.scen_init_flag = False
        
        # record vdi id and index
        self._vdi_id_dict = {}
        self._vpi_id_dict = {}
        state_dict_array, self._vdi_id_dict, self._vpi_id_dict = self._observation_to_ego_dict_array(observation, self.ego_id_list, self._vdi_id_dict, self._vpi_id_dict)

        # record prediction data
        self.ep_prediction_data = {}
        self.ep_prediction_data.update({'ego_gt': {}})

        # record run data for analyse
        if self._record:
            # initialize the record
            self.run_data_dict[self._gt_csv_index] = dict() if self._gt_csv_index not in self.run_data_dict.keys() else self.run_data_dict[self._gt_csv_index]
            self.run_data_dict[self._gt_csv_index][self.ego_id_list[0]] = list() if self.ego_id_list[0] not in self.run_data_dict[self._gt_csv_index].keys() else self.run_data_dict[self._gt_csv_index][self.ego_id_list[0]]
            self.run_data_dict[self._gt_csv_index][self.ego_id_list[0]].append(collections.defaultdict(list))
            # fill the record
            self.run_record = self.run_data_dict[self._gt_csv_index][self.ego_id_list[0]][-1]
            self.run_record['speed'].append(0.)
            self.run_record['completion_rate'].append(0.)
            self.run_record['gt_distance'].append(0.)
            self.run_record['collision'].append(0)
            self.run_record['success'].append(0)

        return state_dict_array

    def step(self, action_dict, prediction=None):
        # send current action to env, consider visualize vehicle trajectory prediction
        content_dict = action_dict
        content_dict.update({'prediction': prediction.tolist() if prediction is not None else prediction})
        message_send = {'command': 'step', 'content': content_dict}
        self._socket.send_string(str(message_send))
        # recieve next observation, reward, done and aux info from env
        message_recv = self._socket.recv()
        message_recv = eval(bytes.decode(message_recv))

        observation = message_recv['observation']
        reward_dict = message_recv['reward']
        done_dict = message_recv['done']
        aux_info_dict = message_recv['aux_info']

        # record prediction data for ade analyse
        if prediction is not None:
            for ego_id in self.ego_id_list:
                # print(self._vdi_id_dict)
                # pridciton in global frame
                prediction_global_index = aux_info_dict[ego_id].pop('prediction_global')
                time = list(prediction_global_index.keys())[0]
                # print(time)
                prediction_global_id = {ego_id: prediction_global_index[time][0]}
                
                for vdi_id in self._vdi_id_dict:
                    vdi_index = int(self._vdi_id_dict[vdi_id][-1])
                    prediction_global_id.update({vdi_id: prediction_global_index[time][vdi_index]})

                self.ep_prediction_data.update({time: prediction_global_id})
                # ego ground truth location
                ego_gt_loc = {time: aux_info_dict[ego_id].pop('ego_loc')}
                self.ep_prediction_data['ego_gt'].update(ego_gt_loc)

            # save file to disk
            all_done = False not in done_dict.values()
            if all_done:
                track_id = aux_info_dict[ego_id].pop('track_id')
                if track_id not in self.prediction_data_dict.keys():
                    self.prediction_data_dict[track_id] = {}
                data_index = len((self.prediction_data_dict[track_id].keys()))
                self.prediction_data_dict[track_id].update({data_index: self.ep_prediction_data})
                with open(self._prediction_filename, 'wb') as f:
                    pickle.dump(self.prediction_data_dict, f)
        
        # record episode data for analyse
        if self._record:        
            # fill the record
            ego_id = self.ego_id_list[0]
            aux_info = aux_info_dict[ego_id]
            self.run_record['speed'].append(aux_info['speed'])
            self.run_record['completion_rate'].append(aux_info['completion_rate'])
            self.run_record['gt_distance'].append(aux_info['distance_to_gt'])
            self.run_record['collision'].append(aux_info['result'] == 'collision')
            self.run_record['success'].append(aux_info['result'] == 'success')
            # save file to disk
            all_done = False not in done_dict.values()
            if all_done:
                with open(self._run_filename, 'wb') as f:
                    pickle.dump(self.run_data_dict, f)
                    print('file saved')
                
        # record vdi id
        state_dict_array, self._vdi_id_dict, self._vpi_id_dict = self._observation_to_ego_dict_array(observation, self.ego_id_list, self._vdi_id_dict, self._vpi_id_dict)
        
        return state_dict_array, reward_dict, done_dict, aux_info_dict

