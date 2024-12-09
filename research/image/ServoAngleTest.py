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

def find_servo_range(pca, channel, file_path):
    """Find the maximum and minimum servo range and save to file."""
    # Create a servo object for the specified channel
    servo = adafruit_motor.servo.Servo(pca.channels[channel])

    # Variables to store the range
    min_position = None
    max_position = None

    try:
        # Sweep the servo to its minimum position
        print("Moving servo to minimum position...")
        servo.angle = 0  # Minimum angle
        time.sleep(2)  # Wait for servo to reach position
        min_position = 0

        # Sweep the servo to its maximum position
        print("Moving servo to maximum position...")
        servo.angle = 180  # Maximum angle
        time.sleep(2)  # Wait for servo to reach position
        max_position = 180

    except ValueError as e:
        print(f"Error occurred while moving servo: {e}")

    # Save the results to a text file
    if min_position is not None and max_position is not None:
        if not os.path.exists(file_path):
            print(f"Creating file: {file_path}")
        else:
            print(f"File already exists. Overwriting: {file_path}")

        with open(file_path, "w") as f:
            f.write(f"Servo Range:\n")
            f.write(f"Minimum Angle: {min_position}\n")
            f.write(f"Maximum Angle: {max_position}\n")

        print(f"Servo range saved to {file_path}")
    else:
        print("Failed to determine servo range.")

if __name__ == "__main__":
    # Initialization
    pca = Servo_Motor_Initialization()

    # Specify the servo channel and file path
    servo_channel = 0  # Change to the appropriate channel for your servo
    output_file_path = "/home/pi/CapstoneProjectBackUp/research/image/servo_range.txt"

    # Find and save the servo range
    find_servo_range(pca, servo_channel, output_file_path)

    # Deinitialize the PCA9685
    pca.deinit()
