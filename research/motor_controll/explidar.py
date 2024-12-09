# #!/usr/bin/env python3
# '''Records scans to a given file
# Usage example:

# $ python3 triallidar.txt scans.py out.txt'''
import sys
import numpy as np
from adafruit_rplidar import RPLidar


PORT_NAME = '/dev/ttyUSB0'

def run(path):
    '''Main function'''
    lidar = RPLidar(None, PORT_NAME, timeout=3)
    data = []
    outfile = open(path, 'w')
    try:
        print('Recording measurments... Press Crl+C to stop.')
        for scan in lidar.iter_scans():
            data.append(np.array(scan))
            outfile.write(str(data) + '\n')
    except KeyboardInterrupt:
        print('Stoping.')
    lidar.stop()
    lidar.disconnect()
    outfile.close()
    
if __name__ == '__main__':
    run(sys.argv[1])
