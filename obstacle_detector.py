#! /usr/bin/env python3

from node import Node
from std_msgs.msg import Int8
from sensor_msgs.msg import Range
from settings import settings


obstacle_detector_publisher = "obstacle_state"
obstacle_detector_subscriber = "obstacle_distance"

obstacle_detector = Node("obstacle_detector_node")


def obsatcle_detection_algorithm(msg: Range):
    distance = msg.range
    result = Int8()
    if distance < settings.OBSTACLE_CLOSE_RANGE:
        result.data = settings.CLOSE_OBSTACLE_CODE
    elif settings.OBSTACLE_CLOSE_RANGE < distance < settings.OBSTACLE_MIDDLE_RANGE:
        result.data = settings.MIDDLE_OBSTACLE_CODE
    else:
        result.data = settings.FAR_OBSTACLE_CODE
    obstacle_detector.publish(obstacle_detector_publisher, result)


def main():
    obstacle_detector.init_publisher(
        obstacle_detector_publisher, "/algo/obstacle", Int8
    )
    obstacle_detector.init_subscriber(
        obstacle_detector_subscriber,
        "raspberry/data/distance",
        Range,
        obsatcle_detection_algorithm,
    )

    obstacle_detector.spin()


main()
