import time

from GPUtil import showUtilization as gpu_usage

if __name__ == '__main__':
    while True:
        print(gpu_usage(all=True, useOldCode=False))
        time.sleep(2)
