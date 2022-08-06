from dis import dis
from node import Node
from std_msgs.msg import Int8
from sensor_msgs.msg import Range
from settings import settings



obstacle_detector_publisher = "obstacle_state"
obstacle_detector_subscriber = "obstacle_distance"

obstacle_detector = Node('obstacle_detector_node')


def obsatcle_detection_algorithm(msg: Range):
    distance = msg.range
    if distance < settings.OBSTACLE_CLOSE_RANGE:
        obstacle_detector.publish(obstacle_detector_publisher, settings.CLOSE_OBSTACLE)
    elif settings.OBSTACLE_CLOSE_RANGE < distance < settings.OBSTACLE_MIDDLE_RANGE:
        obstacle_detector.publish(obstacle_detector_publisher, settings.MIDDLE_OBSTACLE)
    else: 
        obstacle_detector.publish(obstacle_detector_publisher, settings.FAR_OBSTACLE)
        

def main():
    obstacle_detector.init_publisher(obstacle_detector_publisher, "/algo/obstacle", Int8)
    obstacle_detector.init_subscriber(obstacle_detector_subscriber, "raspberry/data/distance", Range,)


main()