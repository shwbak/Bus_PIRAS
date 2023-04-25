import os
import subprocess
from collections import OrderedDict
import glob
import re
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0'(DEBUG), '1'(INFO), '2'(WARNING), '3'(ERROR)}

def getDevEnvInfo():
    dst = ''

    ############################
    # OS Info
    ############################
    os_name = ''
    with open('/etc/issue') as f:
        for line in f:
            index = line.find("\\n")
            os_name = line[:index]
            break

    dst += '{0:15s}{1:s}\n'.format('OS:', os_name)
    # print('{0:15s}{1:s}'.format('OS:', os_name))

    ############################
    # CPU Info
    ############################
    model_name = None
    cpu_core = 0
    with open('/proc/cpuinfo') as f:
        for line in f:
            if line.strip():
                if line.rstrip('\n').startswith('model name'):
                    model_name = line.rstrip('\n').split(':')[1]
                    model_name = model_name.strip()
                if line.rstrip('\n').startswith('cpu cores'):
                    cpu_core = line.rstrip('\n').split(':')[1]

    dst += "{0:15s}{1:s}\n".format('CPU:', model_name)
    # print("{0:15s}{1:s}".format('CPU:', model_name))

    ############################
    # GPU Info
    ############################
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '-q'])
        gpu_info = str(gpu_info)

        index = 0
        gpu_list = []
        while True:
            tmp_index = gpu_info.find('Product Name', index)
            if tmp_index == -1: break
            start_index = gpu_info.find(':', tmp_index)
            start_index += 2
            end_index = gpu_info.find('\\n', tmp_index)
            gpu_name = gpu_info[start_index:end_index]
            gpu_list.append(gpu_name)
            index = end_index

        for i, gpu_name in enumerate(gpu_list):
            dst += '{0:15s}{1:s}\n'.format('GPU_{}:'.format(i), gpu_name)
            # print('{0:15s}{1:s}'.format('GPU_{}:'.format(i), gpu_name))

    except Exception:
        dst += '{0:15s}{1:s}\n'.format('GPU:', 'No GPU')
        # print('{0:15s}{1:s}'.format('GPU:', 'No GPU'))

    ############################
    # Memory Info
    ############################
    meminfo = OrderedDict()
    with open('/proc/meminfo') as f:
        for line in f:
            meminfo[line.split(':')[0]] = line.split(':')[1].strip()
    dst += '{0:15s}{1:s}\n'.format('Memory:', meminfo['MemTotal'])
    # print('{0:15s}{1:s}'.format('Memory:', meminfo['MemTotal']))
    # print('Free memory: {0}'.format(meminfo['MemFree']))

    ############################
    # HDD Info
    ############################
    # Add any other device pattern to read from
    dev_pattern = ['sd.*', 'mmcblk*']

    def size(device):
        nr_sectors = open(device + '/size').read().rstrip('\n')
        sect_size = open(device + '/queue/hw_sector_size').read().rstrip('\n')
        # The sect_size is in bytes, so we convert it to GiB and then send it back
        return (float(nr_sectors) * float(sect_size)) / (1024.0 * 1024.0 * 1024.0)

    for device in glob.glob('/sys/block/*'):
        for pattern in dev_pattern:
            if re.compile(pattern).match(os.path.basename(device)):
                # print('Device:: {0}, Size:: {1:.2f} GiB'.format(device, size(device)))
                dst += '{0:15s}{1:.2f} GiB\n'.format('Storage:', size(device))
                # print('{0:15s}{1:.2f} GiB'.format('Storage:', size(device)))

    ############################
    # Tensorflow Info
    ############################
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        keras_version = tf.keras.__version__
        dst += '{0:15s}{1:s}\n'.format('TensorFlow:', tf_version)
        dst += '{0:15s}{1:s}\n'.format('Keras:', keras_version)
        # print('{0:15s}{1:s}'.format('TensorFlow:', tf_version))
        # print('{0:15s}{1:s}'.format('Keras:', keras_version))
    except:
        dst += '{0:15s}{1:s}\n'.format('TensorFlow:', 'No TensorFlow')
        dst += '{0:15s}{1:s}\n'.format('Keras:', 'No Keras')
        # print('{0:15s}{1:s}'.format('TensorFlow:', 'No TensorFlow'))
        # print('{0:15s}{1:s}'.format('Keras:', 'No Keras'))

    ############################
    # Python Info
    ############################
    python_version = sys.version
    index = python_version.find(' ')
    python_version = python_version[:index]
    dst += '{0:15s}{1:s}\n'.format('Python:', python_version)
    # print('{0:15s}{1:s}'.format('Python:', python_version))

    return dst

if __name__ == '__main__':
    dst = getDevEnvInfo()
    print(dst)
