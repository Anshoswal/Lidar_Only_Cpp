import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ansh/Desktop/lidar_only_cpp/install/my-test-package'
