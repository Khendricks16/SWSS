from gpiozero import OutputDevice
from time import sleep

class Motor():
    """
    Motor control module.

    Contains functions and configuration values related to stepper motor control.

    Author:
        Keith Hendricks
    """

    # Pin Definitions
    IN1 = OutputDevice(14)
    IN2 = OutputDevice(15)
    IN3 = OutputDevice(18)
    IN4 = OutputDevice(23)

    # Define step sequence for the motor
    STEP_SEQUENCE = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]

    @classmethod
    def set_step(cls, w1, w2, w3, w4):
        """
        Sets the output pins to control the motor coils.

        Args:
            w1 (int): State (0 or 1) for coil 1.
            w2 (int): State (0 or 1) for coil 2.
            w3 (int): State (0 or 1) for coil 3.
            w4 (int): State (0 or 1) for coil 4.
        """
        cls.IN1.value = w1
        cls.IN2.value = w2
        cls.IN3.value = w3
        cls.IN4.value = w4

    @classmethod
    def step_motor(cls, steps, direction=1, delay=0.01):
        """
            Moves the powered step motor according to the given values.

            Args:
                steps (int): Number of steps the motor should move.
                direction (int, optional): Direction of movement. Use 1 for forward or -1 for backward. Defaults to 1.
                delay (float, optional): Delay in seconds between each microstep. Adjust to control speed. Defaults to 0.01.

        """
        for _ in range(steps):
            for step in (cls.STEP_SEQUENCE if direction > 0 else reversed(cls.STEP_SEQUENCE)):
                cls.set_step(*step)
                sleep(delay)

