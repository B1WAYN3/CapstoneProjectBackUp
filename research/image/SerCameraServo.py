import os
import math
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
import adafruit_motor.servo
import time

def Servo_Motor_Initialization():
    """Initialize the servo motor using PCA9685."""
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 50  # Set frequency to 50Hz for servos
    return pca

def find_servo_range_and_straight(pca, channel, file_path):
    """Find the servo range and determine the straight angle interactively."""
    # Create a servo object for the specified channel
    servo = adafruit_motor.servo.Servo(pca.channels[channel])

    # Variables to store the range and straight angle
    min_position = None
    max_position = None
    straight_angle = None

    try:
        # Determine the minimum position
        print("Moving servo to minimum position...")
        servo.angle = 0  # Minimum angle
        time.sleep(2)  # Wait for servo to reach position
        min_position = 0
        
        # Prompt to confirm or adjust minimum position
        while True:
            confirm = input("Is this the correct maximum left angle (0)? (y/n): ").lower()
            if confirm == 'y':
                break
            elif confirm == 'n':
                new_min = int(input("Enter a new minimum angle (0-180): "))
                servo.angle = new_min
                time.sleep(2)
                min_position = new_min
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        # Determine the maximum position
        print("Moving servo to maximum position...")
        servo.angle = 180  # Maximum angle
        time.sleep(2)  # Wait for servo to reach position
        max_position = 180

        # Prompt to confirm or adjust maximum position
        while True:
            confirm = input("Is this the correct maximum right angle (180)? (y/n): ").lower()
            if confirm == 'y':
                break
            elif confirm == 'n':
                new_max = int(input("Enter a new maximum angle (0-180): "))
                servo.angle = new_max
                time.sleep(2)
                max_position = new_max
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        # Determine the straight angle
        print("Testing angles to find the straight/forward position...")
        straight_angle = (min_position + max_position) // 2

        while True:
            servo.angle = straight_angle
            time.sleep(2)
            confirm = input(f"Is this the correct straight angle ({straight_angle})? (y/n): ").lower()
            if confirm == 'y':
                break
            elif confirm == 'n':
                straight_angle = int(input("Enter a new straight angle (0-180): "))
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    except ValueError as e:
        print(f"Error occurred while moving servo: {e}")

    # Save the results to a text file
    if min_position is not None and max_position is not None and straight_angle is not None:
        if not os.path.exists(file_path):
            print(f"Creating file: {file_path}")
        else:
            print(f"File already exists. Overwriting: {file_path}")

        with open(file_path, "w") as f:
            f.write(f"Servo Range and Straight Angle:\n")
            f.write(f"Minimum Angle (Left): {min_position}\n")
            f.write(f"Maximum Angle (Right): {max_position}\n")
            f.write(f"Straight Angle: {straight_angle}\n")

        print(f"Servo range and straight angle saved to {file_path}")
    else:
        print("Failed to determine servo range or straight angle.")

if __name__ == "__main__":
    # Initialization
    pca = Servo_Motor_Initialization()

    # Specify the servo channel and file path
    servo_channel = 0  # Change to the appropriate channel for your servo
    output_file_path = "/home/pi/CapstoneProjectBackUp/research/image/servo_camera_range.txt"

    # Find and save the servo range and straight angle
    find_servo_range_and_straight(pca, servo_channel, output_file_path)

    # Deinitialize the PCA9685
    pca.deinit()
