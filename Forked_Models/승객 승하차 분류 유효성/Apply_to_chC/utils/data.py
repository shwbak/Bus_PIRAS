from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from glob import glob
import os
from tqdm import tqdm
from functools import cmp_to_key
import json
import random
import math
import numpy as np
from datetime import datetime
from pytz import timezone
import shutil
from tensorflow import keras
import logging
import traceback
logging.basicConfig(level=logging.ERROR)


class GetOnOffData:
    def __init__(self, original_data_dir, get_on_off_data_dir, dataset_dir, train_val_rate):
        self.original_data_list = self.__getOriginalDataList(original_data_dir)
        self.original_data_dir = original_data_dir
        self.get_on_off_data_dir = get_on_off_data_dir
        self.train_val_rate = train_val_rate
        self.original_num = 0
        self.get_on_json_list = []
        self.get_off_json_list = []
        self.idle_json_list = []
        self.get_on_num = 0
        self.get_off_num = 0
        self.idle_num = 0
        self.dataset_dir = dataset_dir
        self.train_get_on_num = 0
        self.val_get_on_num = 0
        self.test_get_on_num = 0
        self.train_get_off_num = 0
        self.val_get_off_num = 0
        self.test_get_off_num = 0
        self.train_idle_num = 0
        self.val_idle_num = 0
        self.test_idle_num = 0
        self.train_num = 0
        self.val_num = 0
        self.test_num = 0
        self.train_list = []
        self.val_list = []
        self.test_list = []


    def compare_json(self, x, y):
        x_index = [pos for pos, char in enumerate(x) if char == '_']
        x_index = x_index[-1]
        x_index += 1
        y_index = [pos for pos, char in enumerate(y) if char == '_']
        y_index = y_index[-1]
        y_index += 1
        x_int = int(x[x_index:-5])
        y_int = int(y[y_index:-5])
        if (x_int < y_int):
            return -1
        elif (x_int >= y_int):
            return 1

    def __getOriginalDataList(self, original_data_dir):
        # Get Folder List
        folder_list = glob(original_data_dir + '/*/*')
        folder_list = [file for file in folder_list if os.path.isdir(file)]
        folder_list = sorted(folder_list)
        random.seed(0)
        random.shuffle(folder_list)

        # list parameters
        dst = []

        for folder in tqdm(folder_list):
            # Json file list
            json_list = os.listdir(folder)
            json_list = [file for file in json_list if file.endswith('.json')]
            json_list = sorted(json_list, key=cmp_to_key(self.compare_json))

            # Get json path
            tmp_list = []
            for json_file_name in json_list:
                tmp_list.append(os.path.join(folder, json_file_name))

            dst.append(tmp_list)

        return dst

    def getOriginalDataLog(self):
        dst = ''
        json_count = 0
        for folder in self.original_data_list:
            for json_path in folder:
                json_count += 1
                dst += str(json_count) + '. ' + json_path + '\n'
                print(str(json_count) + '. ' + json_path)
        self.original_num = json_count

        return dst

    def makeGetOnOffData(self):
        # Make dir
        get_on_dir = os.path.join(self.get_on_off_data_dir, 'get_on')
        get_off_dir = os.path.join(self.get_on_off_data_dir, 'get_off')
        idle_dir = os.path.join(self.get_on_off_data_dir, 'idle')
        os.makedirs(get_on_dir, exist_ok=True)
        os.makedirs(get_off_dir, exist_ok=True)
        os.makedirs(idle_dir, exist_ok=True)

        # Count Parameters
        get_on_count = 0
        get_off_count = 0
        idle_count = 0

        # file name parameters
        get_on_data_name = 'get_on_'
        get_off_data_name = 'get_off_'
        idle_data_name = 'idle_'

        # Make get on off data
        for json_list in tqdm(self.original_data_list):
            # Get Passenger Data
            for i in range(len(json_list) - 2):
                # Get Json data
                json_path_1 = os.path.join(json_list[i])
                json_path_2 = os.path.join(json_list[i + 1])
                json_path_3 = os.path.join(json_list[i + 2])
                try:
                    with open(json_path_1, 'r') as json_file:
                        json_data_1 = json.load(json_file)
                except Exception:
                    raise Exception('json_path_1: ', json_path_1)
                try:
                    with open(json_path_2, 'r') as json_file:
                        json_data_2 = json.load(json_file)
                except Exception:
                    raise Exception('json_path_2: ', json_path_2)
                try:
                    with open(json_path_3, 'r') as json_file:
                        json_data_3 = json.load(json_file)
                except Exception:
                    raise Exception('json_path_3: ', json_path_3)

                # Get Img width & height
                width = json_data_1['info']['width']
                height = json_data_1['info']['height']

                # Get Valid data id and Append data
                if len(json_data_1['annotations']) != 0 and len(json_data_2['annotations']) != 0 and len(
                        json_data_3['annotations']) != 0:
                    for annotation_1 in json_data_1['annotations']:
                        id_1 = annotation_1['id']
                        for annotation_2 in json_data_2['annotations']:
                            id_2 = annotation_2['id']
                            if id_1 != id_2:
                                continue
                            else:
                                for annotation_3 in json_data_3['annotations']:
                                    id_3 = annotation_3['id']
                                    if id_2 != id_3:
                                        continue
                                    else:
                                        # Checking neck keypoint is exist
                                        if annotation_1['keypoints'][26] == 0 and annotation_2['keypoints'][26] == 0 and \
                                                annotation_3['keypoints'][26] == 0:
                                            break
                                        # Checking head keypoint is exist
                                        elif annotation_1['keypoints'][29] == 0 and annotation_2['keypoints'][
                                            29] == 0 and annotation_3['keypoints'][29] == 0:
                                            break
                                        # Checking right shoulder keypoint is exist
                                        elif annotation_1['keypoints'][38] == 0 and annotation_2['keypoints'][
                                            38] == 0 and annotation_3['keypoints'][38] == 0:
                                            break
                                        # Checking left shoulder keypoint is exist
                                        elif annotation_1['keypoints'][41] == 0 and annotation_2['keypoints'][
                                            41] == 0 and annotation_3['keypoints'][41] == 0:
                                            break

                                        # Get Data of neck keypoint
                                        neck_x_1 = annotation_1['keypoints'][24] / width
                                        neck_y_1 = annotation_1['keypoints'][25] / height
                                        neck_x_2 = annotation_2['keypoints'][24] / width
                                        neck_y_2 = annotation_2['keypoints'][25] / height
                                        neck_x_3 = annotation_3['keypoints'][24] / width
                                        neck_y_3 = annotation_3['keypoints'][25] / height

                                        # Get Data of head keypoint
                                        head_x_1 = annotation_1['keypoints'][27] / width
                                        head_y_1 = annotation_1['keypoints'][28] / height
                                        head_x_2 = annotation_2['keypoints'][27] / width
                                        head_y_2 = annotation_2['keypoints'][28] / height
                                        head_x_3 = annotation_3['keypoints'][27] / width
                                        head_y_3 = annotation_3['keypoints'][28] / height

                                        # Get Data of right shoulder keypoint
                                        r_shoulder_x_1 = annotation_1['keypoints'][36] / width
                                        r_shoulder_y_1 = annotation_1['keypoints'][37] / height
                                        r_shoulder_x_2 = annotation_2['keypoints'][36] / width
                                        r_shoulder_y_2 = annotation_2['keypoints'][37] / height
                                        r_shoulder_x_3 = annotation_3['keypoints'][36] / width
                                        r_shoulder_y_3 = annotation_3['keypoints'][37] / height

                                        # Get Data of left shoulder keypoint
                                        l_shoulder_x_1 = annotation_1['keypoints'][39] / width
                                        l_shoulder_y_1 = annotation_1['keypoints'][40] / height
                                        l_shoulder_x_2 = annotation_2['keypoints'][39] / width
                                        l_shoulder_y_2 = annotation_2['keypoints'][40] / height
                                        l_shoulder_x_3 = annotation_3['keypoints'][39] / width
                                        l_shoulder_y_3 = annotation_3['keypoints'][40] / height

                                        # Get Vector of neck->rShoulder
                                        vector_r_shoulder_x_1 = r_shoulder_x_1 - neck_x_1
                                        vector_r_shoulder_y_1 = r_shoulder_y_1 - neck_y_1
                                        vector_r_shoulder_x_2 = r_shoulder_x_2 - neck_x_2
                                        vector_r_shoulder_y_2 = r_shoulder_y_2 - neck_y_2
                                        vector_r_shoulder_x_3 = r_shoulder_x_3 - neck_x_3
                                        vector_r_shoulder_y_3 = r_shoulder_y_3 - neck_y_3

                                        # Get Vector of neck->lShoulder
                                        vector_l_shoulder_x_1 = l_shoulder_x_1 - neck_x_1
                                        vector_l_shoulder_y_1 = l_shoulder_y_1 - neck_y_1
                                        vector_l_shoulder_x_2 = l_shoulder_x_2 - neck_x_2
                                        vector_l_shoulder_y_2 = l_shoulder_y_2 - neck_y_2
                                        vector_l_shoulder_x_3 = l_shoulder_x_3 - neck_x_3
                                        vector_l_shoulder_y_3 = l_shoulder_y_3 - neck_y_3

                                        # Get Vector of neck->head
                                        vector_head_x_1 = head_x_1 - neck_x_1
                                        vector_head_y_1 = head_y_1 - neck_y_1
                                        vector_head_x_2 = head_x_2 - neck_x_2
                                        vector_head_y_2 = head_y_2 - neck_y_2
                                        vector_head_x_3 = head_x_3 - neck_x_3
                                        vector_head_y_3 = head_y_3 - neck_y_3

                                        # Push Data
                                        data = [[0 for col in range(8)] for row in range(3)]
                                        data[0] = [neck_x_1, neck_y_1,
                                                   vector_head_x_1, vector_head_y_1,
                                                   vector_r_shoulder_x_1, vector_r_shoulder_y_1,
                                                   vector_l_shoulder_x_1, vector_l_shoulder_y_1]
                                        data[1] = [neck_x_2, neck_y_2,
                                                   vector_head_x_2, vector_head_y_2,
                                                   vector_r_shoulder_x_2, vector_r_shoulder_y_2,
                                                   vector_l_shoulder_x_2, vector_l_shoulder_y_2]
                                        data[2] = [neck_x_3, neck_y_3,
                                                   vector_head_x_3, vector_head_y_3,
                                                   vector_r_shoulder_x_3, vector_r_shoulder_y_3,
                                                   vector_l_shoulder_x_3, vector_l_shoulder_y_3]

                                        data_dict = dict()
                                        data_dict['data'] = data
                                        data_dict['original_data'] = []
                                        for i in range(3):
                                            tmp_dict = dict()
                                            tmp_dict['time_step'] = i + 1
                                            if i == 0:
                                                tmp_dict['original_data_path'] = json_path_1
                                            elif i == 1:
                                                tmp_dict['original_data_path'] = json_path_2
                                            else:
                                                tmp_dict['original_data_path'] = json_path_3
                                            tmp_dict['id'] = annotation_3['id']

                                            data_dict['original_data'].append(tmp_dict)


                                        # Count num of get_on get_off idle
                                        if annotation_3['get_on']:
                                            get_on_count += 1
                                            data_dict['label'] = 0
                                            json_file_name = get_on_data_name + str(get_on_count) + '.json'
                                            json_file_path = os.path.join(get_on_dir, json_file_name)
                                            if not os.path.exists(json_file_path):
                                                with open(json_file_path, 'w') as f:
                                                    json.dump(data_dict, f, indent=4)

                                        elif annotation_3['get_off']:
                                            get_off_count += 1
                                            data_dict['label'] = 1
                                            json_file_name = get_off_data_name + str(get_off_count) + '.json'
                                            json_file_path = os.path.join(get_off_dir, json_file_name)
                                            if not os.path.exists(json_file_path):
                                                with open(json_file_path, 'w') as f:
                                                    json.dump(data_dict, f, indent=4)

                                        else:
                                            idle_count += 1
                                            data_dict['label'] = 2
                                            json_file_name = idle_data_name + str(idle_count) + '.json'
                                            json_file_path = os.path.join(idle_dir, json_file_name)
                                            if not os.path.exists(json_file_path):
                                                with open(json_file_path, 'w') as f:
                                                    json.dump(data_dict, f, indent=4)

                                        break
                            break

        self.get_on_num = get_on_count
        self.get_off_num = get_off_count
        self.idle_num = idle_count

    def getGetOnDataLog(self):
        dst = ''

        # Get dir
        get_on_dir = os.path.join(self.get_on_off_data_dir, 'get_on')

        # Get json list
        json_list = glob(get_on_dir + '/*')
        json_list = sorted(json_list, key=cmp_to_key(self.compare_json))
        self.get_on_json_list = json_list

        for i in range(self.get_on_num):
            tmp_str = str(i + 1) + '. ' + json_list[i] + '\n'
            with open(json_list[i], 'r') as f:
                json_data = json.load(f)
            for original_data_dict in json_data['original_data']:
                time_step = original_data_dict['time_step']
                original_data_path = original_data_dict['original_data_path']
                id = original_data_dict['id']
                tmp_str += '\ttime_step: {0:d}, original_data_path: {1:s}, id: {2:d}\n'.format(time_step, original_data_path, id)
            dst += tmp_str
            print(tmp_str)

        return dst

    def getGetOffDataLog(self):
        dst = ''

        # Get dir
        get_off_dir = os.path.join(self.get_on_off_data_dir, 'get_off')

        # Get json list
        json_list = glob(get_off_dir + '/*')
        json_list = sorted(json_list, key=cmp_to_key(self.compare_json))
        self.get_off_json_list = json_list

        for i in range(self.get_off_num):
            tmp_str = str(i + 1) + '. ' + json_list[i] + '\n'
            with open(json_list[i], 'r') as f:
                json_data = json.load(f)
            for original_data_dict in json_data['original_data']:
                time_step = original_data_dict['time_step']
                original_data_path = original_data_dict['original_data_path']
                id = original_data_dict['id']
                tmp_str += '\ttime_step: {0:d}, original_data_path: {1:s}, id: {2:d}\n'.format(time_step, original_data_path, id)
            dst += tmp_str
            print(tmp_str)

        return dst

    def getIdleDataLog(self):
        dst = ''

        # Get dir
        idle_dir = os.path.join(self.get_on_off_data_dir, 'idle')

        # Get json list
        json_list = glob(idle_dir + '/*')
        json_list = sorted(json_list, key=cmp_to_key(self.compare_json))
        self.idle_json_list = json_list

        for i in range(self.idle_num):
            tmp_str = str(i + 1) + '. ' + json_list[i] + '\n'
            with open(json_list[i], 'r') as f:
                json_data = json.load(f)
            for original_data_dict in json_data['original_data']:
                time_step = original_data_dict['time_step']
                original_data_path = original_data_dict['original_data_path']
                id = original_data_dict['id']
                tmp_str += '\ttime_step: {0:d}, original_data_path: {1:s}, id: {2:d}\n'.format(time_step, original_data_path, id)
            dst += tmp_str
            print(tmp_str)

        return dst

    def divideDataByRate(self):
        # Get train/val/test dir and make dir
        train_dir = os.path.join(self.dataset_dir, 'training')
        val_dir = os.path.join(self.dataset_dir, 'validation')
        test_dir = os.path.join(self.dataset_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Divide data by rate
        self.train_get_on_num = math.ceil(self.get_on_num * self.train_val_rate[0])  # 올림
        self.val_get_on_num = math.floor(self.get_on_num * self.train_val_rate[1])  # 내림
        self.test_get_on_num = self.get_on_num - self.train_get_on_num - self.val_get_on_num

        self.train_get_off_num = math.ceil(self.get_off_num * self.train_val_rate[0])  # 올림
        self.val_get_off_num = math.floor(self.get_off_num * self.train_val_rate[1])  # 내림
        self.test_get_off_num = self.get_off_num - self.train_get_off_num - self.val_get_off_num

        self.train_idle_num = math.ceil(self.idle_num * self.train_val_rate[0])  # 올림
        self.val_idle_num = math.floor(self.idle_num * self.train_val_rate[1])  # 내림
        self.test_idle_num = self.idle_num - self.train_idle_num - self.val_idle_num

        self.train_num = self.train_get_on_num + self.train_get_off_num + self.train_idle_num
        self.val_num = self.val_get_on_num + self.val_get_off_num + self.val_idle_num
        self.test_num = self.test_get_on_num + self.test_get_off_num + self.test_idle_num

        # Get train/val/test list
        train_json_list = self.get_on_json_list[:self.train_get_on_num] + \
                          self.get_off_json_list[:self.train_get_off_num] + \
                          self.idle_json_list[:self.train_idle_num]

        val_json_list = self.get_on_json_list[self.train_get_on_num:self.train_get_on_num + self.val_get_on_num] + \
                        self.get_off_json_list[self.train_get_off_num:self.train_get_off_num + self.val_get_off_num] + \
                        self.idle_json_list[self.train_idle_num:self.train_idle_num + self.val_idle_num]

        test_json_list = self.get_on_json_list[self.train_get_on_num + self.val_get_on_num:] + \
                         self.get_off_json_list[self.train_get_off_num + self.val_get_off_num:] + \
                         self.idle_json_list[self.train_idle_num + self.val_idle_num:]

        # Copy get_on/get_off/idle json files to train/validation/test dataset dir
        for json_path in tqdm(train_json_list):
            new_json_path = os.path.join(train_dir, os.path.basename(json_path))
            self.train_list.append(new_json_path)
            if not os.path.exists(new_json_path):
                shutil.copyfile(json_path, new_json_path)
        for json_path in tqdm(val_json_list):
            new_json_path = os.path.join(val_dir, os.path.basename(json_path))
            self.val_list.append(new_json_path)
            if not os.path.exists(new_json_path):
                shutil.copyfile(json_path, new_json_path)
        for json_path in tqdm(test_json_list):
            new_json_path = os.path.join(test_dir, os.path.basename(json_path))
            self.test_list.append(new_json_path)
            if not os.path.exists(new_json_path):
                shutil.copyfile(json_path, new_json_path)

    def getTrainDataLog(self):
        dst = ''
        for idx in range(self.train_num):
            json_path = self.train_list[idx]
            dst += '{0:d}. {1:s}'.format(idx + 1, json_path) + '\n'
            print('{0:d}. {1:s}'.format(idx + 1, json_path))
        return dst

    def getValDataLog(self):
        dst = ''
        for idx in range(self.val_num):
            json_path = self.val_list[idx]
            dst += '{0:d}. {1:s}'.format(idx + 1, json_path) + '\n'
            print('{0:d}. {1:s}'.format(idx + 1, json_path))
        return dst

    def getTestDataLog(self):
        dst = ''
        for idx in range(self.test_num):
            json_path = self.test_list[idx]
            dst += '{0:d}. {1:s}'.format(idx + 1, json_path) + '\n'
            print('{0:d}. {1:s}'.format(idx + 1, json_path))
        return dst

    def generateGetOnOffData(self):
        # Parameters
        fmt = '%Y-%m-%d %H:%M:%S %Z%z'
        log_str = ''

        # Get Original Data Log str
        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' 원본 데이터 파일 목록'
        log_str += tmp_str + '\n'
        print(tmp_str)

        original_data_log_str = self.getOriginalDataLog()
        log_str += original_data_log_str + '\n'

        # Start making get_on, get_off, idle data
        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' 승하차 데이터 가공 시작'
        log_str += tmp_str + '\n'
        print(tmp_str)

        self.makeGetOnOffData()

        # Get get_on data list
        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' 승차(Get On) 데이터 파일 목록 및 사용된 원본 데이터 목록'
        log_str += tmp_str + '\n'
        print(tmp_str)

        get_on_log_str = self.getGetOnDataLog()
        log_str += get_on_log_str + '\n'

        # Get get_off data list
        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' 하차(Get Off) 데이터 파일 목록 및 사용된 원본 데이터 목록'
        log_str += tmp_str + '\n'
        print(tmp_str)

        get_off_log_str = self.getGetOffDataLog()
        log_str += get_off_log_str + '\n'

        # Get Idle data list
        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' 무관(Idle) 데이터 파일 목록 및 사용된 원본 데이터 목록'
        log_str += tmp_str + '\n'
        print(tmp_str)

        idle_log_str = self.getIdleDataLog()
        log_str += idle_log_str + '\n'

        # Start divide training, validation, test data
        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' Training(80%)/Validation(10%)/Test(10%) 데이터 분할 시작'
        log_str += tmp_str + '\n'
        print(tmp_str)

        self.divideDataByRate()

        # Get train data list
        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' Training(80%) 데이터 파일 목록'
        log_str += tmp_str + '\n'
        print(tmp_str)

        train_log_str = self.getTrainDataLog()
        log_str += train_log_str + '\n'

        # Get val data list
        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' Validation(10%) 데이터 파일 목록'
        log_str += tmp_str + '\n'
        print(tmp_str)

        val_log_str = self.getValDataLog()
        log_str += val_log_str + '\n'

        # Get test data list
        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' Test(10%) 데이터 파일 목록'
        log_str += tmp_str + '\n'
        print(tmp_str)

        test_log_str = self.getTestDataLog()
        log_str += test_log_str + '\n'

        # summary of data
        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' 데이터 요약'
        log_str += tmp_str + '\n'
        print(tmp_str)

        tmp_str = '원본 데이터 JSON 파일 개수: {}'.format(str(self.original_num))
        log_str += tmp_str + '\n'
        print(tmp_str)

        tmp_str = '승차(Get On) 데이터 JSON 파일 개수: {}'.format(str(self.get_on_num))
        log_str += tmp_str + '\n'
        print(tmp_str)

        tmp_str = '하차(Get Off) 데이터 JSON 파일 개수: {}'.format(str(self.get_off_num))
        log_str += tmp_str + '\n'
        print(tmp_str)

        tmp_str = '무관(Idle) 데이터 JSON 파일 개수: {}'.format(str(self.idle_num))
        log_str += tmp_str + '\n'
        print(tmp_str)

        tmp_str = 'Training(80%) 데이터 JSON 파일 개수: {} (Get On: {}, Get Off: {}, Idle: {})'.format(
            self.train_num,
            self.train_get_on_num,
            self.train_get_off_num,
            self.train_idle_num
        )
        log_str += tmp_str + '\n'
        print(tmp_str)

        tmp_str = 'Validation(10%) 데이터 JSON 파일 개수: {} (Get On: {}, Get Off: {}, Idle: {})'.format(
            self.val_num,
            self.val_get_on_num,
            self.val_get_off_num,
            self.val_idle_num
        )
        log_str += tmp_str + '\n'
        print(tmp_str)

        tmp_str = 'Test(10%) 데이터 JSON 파일 개수: {} (Get On: {}, Get Off: {}, Idle: {})'.format(
            self.test_num,
            self.test_get_on_num,
            self.test_get_off_num,
            self.test_idle_num
        )
        log_str += tmp_str + '\n'
        print(tmp_str)

        cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
        timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
        tmp_str = timestamp_str + ' 데이터 가공 완료'
        log_str += tmp_str + '\n\n'
        print(tmp_str)

        return log_str


class GetOnOffDataset:
    def __init__(self, dataset_dir):
        self.train_list = []
        self.val_list = []
        self.test_list = []
        self.__getDatasetList(dataset_dir)

    def compare_json(self, x, y):
        x_index = [pos for pos, char in enumerate(x) if char == '_']
        x_index = x_index[-1]
        x_index += 1
        y_index = [pos for pos, char in enumerate(y) if char == '_']
        y_index = y_index[-1]
        y_index += 1
        x_int = int(x[x_index:-5])
        y_int = int(y[y_index:-5])
        if (x_int < y_int):
            return -1
        elif (x_int >= y_int):
            return 1

    def __getSortedDataList(self, data_list):
        get_on_list = [file for file in data_list if os.path.basename(file).find('get_on') != -1]
        get_on_list = sorted(get_on_list, key=cmp_to_key(self.compare_json))
        get_off_list = [file for file in data_list if os.path.basename(file).find('get_off') != -1]
        get_off_list = sorted(get_off_list, key=cmp_to_key(self.compare_json))
        idle_list = [file for file in data_list if os.path.basename(file).find('idle') != -1]
        idle_list = sorted(idle_list, key=cmp_to_key(self.compare_json))

        dst = get_on_list + get_off_list + idle_list

        return dst

    def __getDatasetList(self, dataset_dir):
        train_dir = os.path.join(dataset_dir, 'training')
        val_dir = os.path.join(dataset_dir, 'validation')
        test_dir = os.path.join(dataset_dir, 'test')

        train_data_list = glob(train_dir + '/*')
        val_data_list = glob(val_dir + '/*')
        test_data_list = glob(test_dir + '/*')

        self.train_list = self.__getSortedDataList(train_data_list)
        self.val_list = self.__getSortedDataList(val_data_list)
        self.test_list = self.__getSortedDataList(test_data_list)

    def getTrainDatasetNumpyArray(self):
        dst_x = []
        dst_y = []

        for json_path in tqdm(self.train_list):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            data = json_data['data']
            label = json_data['label']
            dst_x.append(data)
            dst_y.append(label)

        dst_x = np.array(dst_x, dtype=np.float32)
        dst_y = keras.utils.to_categorical(dst_y, num_classes=3)
        dst_y = np.array(dst_y, dtype=np.float32)

        return dst_x, dst_y

    def getValDatasetNumpyArray(self):
        dst_x = []
        dst_y = []

        for json_path in tqdm(self.val_list):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            data = json_data['data']
            label = json_data['label']
            dst_x.append(data)
            dst_y.append(label)

        dst_x = np.array(dst_x, dtype=np.float32)
        dst_y = keras.utils.to_categorical(dst_y, num_classes=3)
        dst_y = np.array(dst_y, dtype=np.float32)

        return dst_x, dst_y

    def getTestDatasetNumpyArray(self):
        dst_x = []
        dst_y = []

        for json_path in tqdm(self.test_list):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            data = json_data['data']
            label = json_data['label']
            dst_x.append(data)
            dst_y.append(label)

        dst_x = np.array(dst_x, dtype=np.float32)
        dst_y = keras.utils.to_categorical(dst_y, num_classes=3)
        dst_y = np.array(dst_y, dtype=np.float32)

        return dst_x, dst_y


if __name__ == '__main__':
    fmt = '%Y-%m-%d %H:%M:%S %Z%z'
    cur_dir = os.getcwd()
    original_data_dir = os.path.join(cur_dir, '../original_data/General/A')
    get_on_off_data_dir = os.path.join(cur_dir, '../get_on_off_data')
    dataset_dir = os.path.join(cur_dir, '../dataset')
    train_val_rate = (0.8, 0.1)
    f = open('../log.txt', 'w')

    # Get Data Instance
    cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
    timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
    tmp_str = timestamp_str + ' 데이터 가공 시작'
    log_str = tmp_str + '\n'
    f.write(log_str)
    print(tmp_str)

    get_on_off_data = GetOnOffData(original_data_dir, get_on_off_data_dir, dataset_dir, train_val_rate)
    log_str = get_on_off_data.generateGetOnOffData()
    f.write(log_str)

    # Start make train/validation/test numpy array
    cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
    timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
    tmp_str = timestamp_str + ' Training/Validation/Test 데이터 numpy 배열로 가공 시작'
    log_str = tmp_str + '\n'
    f.write(log_str)
    print(tmp_str)

    get_on_off_dataset = GetOnOffDataset(dataset_dir)

    # Get train data list
    cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
    timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
    tmp_str = timestamp_str + ' Training(80%) 데이터 numpy 배열 정보'
    log_str = tmp_str + '\n'
    f.write(log_str)
    print(tmp_str)

    train_x, train_y = get_on_off_dataset.getTrainDatasetNumpyArray()
    log_str = ''
    log_str += 'Train_X Shape: {}\n'.format(train_x.shape)
    log_str += 'Train_Y Shape: {}\n'.format(train_y.shape)
    f.write(log_str)
    print(log_str)

    # Get val data list
    cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
    timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
    tmp_str = timestamp_str + ' Validation(10%) 데이터 numpy 배열 정보'
    log_str = tmp_str + '\n'
    f.write(log_str)
    print(tmp_str)
    val_x, val_y = get_on_off_dataset.getValDatasetNumpyArray()
    log_str = ''
    log_str += 'Val_X Shape: {}\n'.format(val_x.shape)
    log_str += 'Val_Y Shape: {}\n'.format(val_y.shape)
    f.write(log_str)
    print(log_str)

    # Get test data list
    cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
    timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
    tmp_str = timestamp_str + ' Test(10%) 데이터 numpy 배열 정보'
    log_str = tmp_str + '\n'
    f.write(log_str)
    print(tmp_str)
    test_x, test_y = get_on_off_dataset.getTestDatasetNumpyArray()
    log_str = ''
    log_str += 'Test_X Shape: {}\n'.format(test_x.shape)
    log_str += 'Test_Y Shape: {}\n'.format(test_y.shape)
    f.write(log_str)
    print(log_str)

    f.close()
