import math
import time
import busio
from math import cos, sin, pi
from board import SCL, SDA
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from adafruit_rplidar import RPLidar

def Servo_Motor_Initialization():
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 100
    return pca

def Motor_Speed(pca, percent):
    speed = ((percent) * 3277) + 65535 * 0.15
    pca.channels[15].duty_cycle = math.floor(speed)
    print(f'Motor speed set to {speed / 65535:.2f}')

# Initialization
pca = Servo_Motor_Initialization()
servo = servo.Servo(pca.channels[14]) 
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME, timeout=3)

def update_steering_angle(angle):
    servo.angle = angle

def scale_lidar_distance(distance, max_distance=3000):
    return min(distance, max_distance) / max_distance

def main():
    update_steering_angle(90)  # Start with neutral steering angle
    Motor_Speed(pca, 0.24)  # Start with motor at 0.24
    # object_detected_behind = False  # Flag to detect objects behind the vehicle

    try:
        while True:
            # object_detected_behind = False  # Reset flag each cycle
            for scan in lidar.iter_scans():
                for (_, angle, distance) in scan:
                    angle = int(angle)
                    # print(f"Distance: {distance} mm, Angle: {angle} degrees")

                    # Detect objects around the vehicle
                    if distance <= 350 and (angle in range(315, 360) or angle in range(0, 45)):
                        print(f"Object detected behind at {distance} mm and {angle} degrees.")
                        # Motor_Speed(pca, 0.25)
                        # object_detected_behind = True  # Set flag when object is detected
                    elif distance <= 800 and (angle in range(165, 205)):
                        print(f"Object is in front of us at {distance} mm and {angle} degrees. slowing down")
                        Motor_Speed(pca, -1)
                        time.sleep(2)
                        # Motor_Speed(pca,0)
                    elif distance <= 400 and (angle in range(45, 180)):
                        print(f"Object is on the left at {distance} mm and {angle} degrees., move right")
                        update_steering_angle(60)
                    elif distance <= 400 and (angle in range(230, 315)):
                        print(f"Object is on the right at {distance} mm and {angle} degrees., move left")
                        update_steering_angle(125)

               #  if not object_detected_behind:
               #      print("No object detected behind, stopping motor.")
               #      Motor_Speed(pca, 0)

    except KeyboardInterrupt:
        print('Stopping due to keyboard interrupt.')
        Motor_Speed(pca, 0)  
    finally:
        lidar.stop()
        lidar.disconnect()

if __name__ == "__main__":
    main()
