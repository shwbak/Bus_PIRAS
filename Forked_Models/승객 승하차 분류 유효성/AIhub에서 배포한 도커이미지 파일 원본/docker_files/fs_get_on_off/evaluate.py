from __future__ import print_function
from __future__ import absolute_import

######################################
# Import Library
######################################
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import os
from pytz import timezone
import numpy as np
from utils.dev_env_info import getDevEnvInfo
from utils.data import GetOnOffDataset
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix


######################################
# Set GPU
######################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


######################################
# Set GPU Memory
######################################
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
#   except RuntimeError as e:
#     print(e)


######################################
# Set Functions
######################################
def getConfusionMatrixLog(confusion_matrix_array, class_name):
    dst = ''
    dst += '{0:15s}{1:15s}{2:s}\n'.format('', '', 'Predicted Class')
    for idx in range(len(class_name) + 1):
        if idx == 0:
            dst += '{0:15s}{1:15s}'.format('', '')
            for name in class_name:
                dst += '{0:15s}'.format(name)

        else:
            if idx == 1:
                dst += '\n{0:15s}'.format('True Class')
                dst += '{0:15s}'.format(class_name[idx - 1])

            else:
                dst += '{0:15s}{1:15s}'.format('', class_name[idx - 1])

            for value in confusion_matrix_array[idx - 1]:
                dst += '{0:15s}'.format(str(value))
            dst += '\n'

    return dst


def getClassReportLog(class_report_dict, class_name):
    dst = ''
    dst += '{0:15s}{1:15s}{2:15s}{3:15s}\n'.format('', 'Precision', 'Recall', 'F1-score')
    for name in class_name:
        precision = class_report_dict[name]['precision']
        precision = '{0:.2f}'.format(precision)
        recall = class_report_dict[name]['recall']
        recall = '{0:.2f}'.format(recall)
        f1_score = class_report_dict[name]['f1-score']
        f1_score = '{0:.2f}'.format(f1_score)
        dst += '{0:15s}{1:15s}{2:15s}{3:15s}\n'.format(name,
                                                       precision,
                                                       recall,
                                                       f1_score)

    precision = class_report_dict['weighted avg']['precision']
    precision = '{0:.2f}'.format(precision)
    recall = class_report_dict['weighted avg']['recall']
    recall = '{0:.2f}'.format(recall)
    f1_score = class_report_dict['weighted avg']['f1-score']
    f1_score = '{0:.2f}'.format(f1_score)
    dst += '\n{0:15s}{1:15s}{2:15s}{3:15s}\n'.format('Total',
                                                     precision,
                                                     recall,
                                                     f1_score)

    return dst


######################################
# Global Parameters
######################################
fmt = '%Y-%m-%d %H:%M:%S %Z%z'
cur_dir = os.getcwd()
dataset_dir = os.path.join(cur_dir, 'dataset')
model_save_dir = os.path.join(cur_dir, 'saved_models')
f = open('output/evaluate_log.txt', 'w')
class_name = ['Get On', 'Get Off', 'Idle']


######################################
# Get Development Environment Info
######################################
cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
tmp_str = timestamp_str + ' 개발 환경 정보'
log_str = tmp_str + '\n'
f.write(log_str)
print(tmp_str)

tmp_str = getDevEnvInfo()
log_str = tmp_str + '\n'
f.write(log_str)
print(tmp_str)


######################################
# Get Test Data
######################################
# Start make test numpy array
cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
tmp_str = timestamp_str + ' Test 데이터 정보 가져오기 시작'
log_str = tmp_str + '\n'
f.write(log_str)
print(tmp_str)

get_on_off_dataset = GetOnOffDataset(dataset_dir)
test_list = get_on_off_dataset.test_list

# Get test data list
cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
tmp_str = timestamp_str + ' Test 데이터 목록'
log_str = tmp_str + '\n'
f.write(log_str)
print(tmp_str)

tmp_str = ''
for idx, test_json_path in enumerate(test_list):
    tmp_str += str(idx + 1) + '. ' + test_json_path + '\n'
    print(test_json_path)
tmp_str += '\n'
f.write(tmp_str)

# Get test numpy array
cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
tmp_str = timestamp_str + ' Test 데이터 numpy 배열 정보'
log_str = tmp_str + '\n'
f.write(log_str)
print(tmp_str)
test_x, test_y = get_on_off_dataset.getTestDatasetNumpyArray()
log_str = ''
log_str += 'Test_X Shape: {}\n'.format(test_x.shape)
log_str += 'Test_Y Shape: {}\n'.format(test_y.shape)
log_str += '\n'
f.write(log_str)
print(log_str)

# Get Test Dataset API
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_dataset = test_dataset.batch(1)


######################################
# Get Model
######################################
cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
tmp_str = timestamp_str + ' 훈련 완료된 베스트 모델'
log_str = tmp_str + '\n'
f.write(log_str)
print(tmp_str)

model_list = glob(model_save_dir + '/*')
model_list = sorted(model_list)
model_path = model_list[-1]
tmp_str = model_path + '\n\n'
f.write(tmp_str)
print(tmp_str)

model = keras.models.load_model(model_path)

######################################
# Start Evaluate
######################################
cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
tmp_str = timestamp_str + ' 모델 평가 시작'
log_str = tmp_str + '\n'
f.write(log_str)
print(tmp_str)

y_true = []
y_pred = []
for idx, (input_x, input_y) in enumerate(test_dataset.as_numpy_iterator()):
    label = np.argmax(input_y, axis=1)
    prediction = np.argmax(model(input_x, training=False), axis=1)

    cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
    timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
    tmp_str = timestamp_str + ' ' + str(idx+1) + '. ' + test_list[idx] + ' Test 데이터 인풋 결과'
    log_str = tmp_str + '\n'
    f.write(log_str)
    print(tmp_str)

    tmp_str = ''
    label_name = ''
    pred_name = ''

    if label == 0:
        label_name = 'Get On'
    elif label == 1:
        label_name = 'Get Off'
    else:
        label_name = 'Idle'

    if prediction == 0:
        pred_name = 'Get On'
    elif prediction == 1:
        pred_name = 'Get Off'
    else:
        pred_name = 'Idle'

    tmp_str += '정답: {}\n'.format(label_name)
    tmp_str += '예측: {}\n'.format(pred_name)
    f.write(tmp_str)
    print(tmp_str)

    y_true += list(label)
    y_pred += list(prediction)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
tmp_str = timestamp_str + ' 전체 Test 데이터 종합 평가 시작'
log_str = '\n' + tmp_str + '\n'
f.write(log_str)
print(tmp_str)

# Get Confusion Matrix
cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
tmp_str = timestamp_str + ' 전체 Test 데이터 종합 혼동 행렬 결과'
log_str = tmp_str + '\n'
f.write(log_str)
print(tmp_str)

confusion_matrix_array = confusion_matrix(y_true, y_pred)

log_str = getConfusionMatrixLog(confusion_matrix_array, class_name)
log_str += '\n'
f.write(log_str)
print(log_str)

cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
tmp_str = timestamp_str + ' 전체 Test 데이터 종합 결과'
log_str = tmp_str + '\n'
f.write(log_str)
print(tmp_str)

class_report_dict = classification_report(y_true, y_pred, target_names=class_name, output_dict=True)

log_str = getClassReportLog(class_report_dict, class_name)
log_str += '\n'
f.write(log_str)
print(log_str)

cur_time = datetime.now(timezone('Asia/Seoul')).strftime(fmt)
timestamp_str = '{0:s}'.format('[Timestamp: ' + cur_time + ']')
tmp_str = timestamp_str + ' 모델 평가 종료'
log_str = tmp_str + '\n'
f.write(log_str)
print(tmp_str)

f.close()
