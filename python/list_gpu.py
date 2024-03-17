import tensorflow as tf;
from tensorflow.python.platform import build_info as tf_build_info

for physical_device in tf.config.list_physical_devices('GPU') :
    print(tf.config.experimental.get_device_details(physical_device))

print("cudnn_version",tf_build_info.build_info['cudnn_version'])
print("cuda_version",tf_build_info.build_info['cuda_version'])