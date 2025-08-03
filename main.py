"""
Main python script that will be start program on RPi for the
SWSS project (for FEDD).

Author:
    Keith Hendricks
"""

# RPI libraries
from picamzero import Camera
import Motor

# AI libraries
import WasteNet

# Built in python libraries
import logging
import os
import time
import sys

# Logger set up
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"./logs/temperature.log",
    encoding="utf-8",
    level=logging.INFO,
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%I:%M:%S %m/%d/%Y"
)

def main():
    """
    Main function to start the program.
        
    This function serves as the entry point for the script.
    """
    # Define place that camera captures will be stored
    CAPTURES_LOCATION = './assets/captures/'

    # Set up camera
    cam = Camera()

    # START OF SORTING ACTION
    print("\n\n============== START OF PROGRAM ==============\n\n")
    logger.info("Program started")

    # Take photo in 10 seconds
    for i in range(10, 0, -1):
        print(f"\rTaking photo in {i}...", end='')
        sys.stdout.flush()
        time.sleep(1)

    print("\rTaking photo now!     ")  # Clear the line afterward
    photo_destination = f"{CAPTURES_LOCATION}/new_waste_image.jpg"
    cam.take_photo(photo_destination)

    # Categorize photo
    print("\nCategorizing ...")
    time.sleep(5)
    found_category = WasteNet.predict(photo_destination)
    print(f"Category: [{found_category}]")
    logger.info("Categorization {found_category} determined")
    time.sleep(5)


    # Move motor appropriately
    Motor.step_motor(500, 1)

    # Showcase that end of program has been reached
    print("\n\n============== END OF PROGRAM ==============\n\n")

    # Wait a bit and then reset motors
    time.sleep(3)
    Motor.step_motor(500, -1)

    logger.info("Program finished")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped by user\n")
