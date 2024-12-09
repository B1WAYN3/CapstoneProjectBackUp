#Test_lidar
import math
import os
import time
import busio
from math import cos, sin, pi, floor
from board import SCL, SDA
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from adafruit_rplidar import RPLidar
import adafruit_motor.servo

def Servo_Motor_Initialization():
   i2c_bus = busio.I2C(SCL,SDA)
   pca = PCA9685(i2c_bus)
   pca.frequency = 100
   return pca
   
def Motor_Speed(pca,percent):
   #converts a -1 to 1 value to 16-bit duty cycle
   speed = ((percent) * 3277) + 65535 * 0.15
   pca.channels[15].duty_cycle = math.floor(speed)
   print(speed/65535)

#initialization
pca = Servo_Motor_Initialization()

Motor_Speed(pca,0)

# i2c = busio.I2C(SCL, SDA)
# pca = PCA9685(i2c)
# pca.frequency = 100

# os.putenv('SDL_FBDEV', '/dev/fb1')
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME, timeout=3)
steering_channel = 14
motor_channel = 15
servo_steering = servo.Servo(pca.channels[steering_channel])

def update_steering_angle(angle):
    servo_steering.angle = angle
    
def scale_lidar_distance(distance, max_distance=3000):
    return min(distance, max_distance) / max_distance

def main():
    # neutral
    update_steering_angle(90)
    #time.sleep(0.1)
    Motor_Speed(pca,-0.225)
    stop_motor = False
    reset = False
    
    try:
        scan_data = [0]*360
        while True:
            for scan in lidar.iter_scans():
                for (_, angle, distance) in scan:
                    angle = int(angle)
                    
                    # Print Out Data
                    f_dist = f"{distance:.2f}"
                    f_angle = f"{angle:.2f}"
                    print(f"D: {f_dist} mm, A: {f_angle} deg")
                    
                    #Motor_Speed(pca,0.225)# if stop_motor:
                    #     continue
                    # Back
                    if distance <= 350 and (angle in range(315, 360) or angle in range(0,45)):
                        print("Object is behind us, speed up!")
                        Motor_Speed(pca,0.225)
                    # else:
                    #     Motor_Speed(pca,0.1)


                        # time.sleep(1)
                        # exit()
                        
                    # Front
                    if distance <= 600 and (angle in range(150, 215)):
                        print("Object is in front of us, slow down")
                        Motor_Speed(pca,0)
                        # time.sleep(1)
                        # stop_motor = True
                        # exit()
                        
                    # Left
                    if distance <= 200 and (angle in range(45, 180)):
                        print("Object is on the left, move right")
                        update_steering_angle(60)
                        
                    # Right
                    if distance <= 200 and (angle in range(230, 315)):
                        print("Object is on the right, move left")
                        update_steering_angle(135)
                                            
                # LIDAR scaling
                for angle in range(360):
                    distance = scan_data[angle]
                    if distance:
                        scaled_distance = scale_lidar_distance(distance)
                        radians = angle * pi / 180
                        x = scaled_distance * cos(radians) * 119
                        y = scaled_distance * sin(radians) * 119
                        point = (160 + int(x), 120 + int(y))
                        
    except KeyboardInterrupt:
        print('Stopping.')
        Motor_Speed(pca,0.1)
    finally:
        lidar.stop()
        lidar.disconnect()
if __name__ == "__main__":
    main()
