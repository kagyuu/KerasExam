import tensorflow as tf;

for physical_device in tf.config.list_physical_devices('GPU') :
    print(tf.config.experimental.get_device_details(physical_device))
