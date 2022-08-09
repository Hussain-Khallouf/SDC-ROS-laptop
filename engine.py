#! /usr/bin/env python3

from deps import get_map_handler
from node import Node
from std_msgs.msg import String, Int8
from settings import settings



map_handler = get_map_handler()


obstacle_detector_subscriber = "obstacle_detector_subscriber"
lane_keeping_assistant_subscriber = "lane_keeping_assistant_subscriber"
lane_detection_subscriber = "lane_detection_subscriber"

engine = Node("engine_node")


def obstacle_avoiding_cd(msg: Int8):
    range = msg.data
    command = ""
    if range == settings.CLOSE_OBSTACLE:
        command = settings.STOP_ENGINE_COMMAND
    else :
        command = settings.GO_ENGINE_COMMAND
    engine.publish("commands", command)
    print(f"angle: {range}")


def main():
    engine.init_publisher("commands", "engine/commands", String)
    # engine.init_subscriber(
    #     obstacle_detector_subscriber,
    #     "lane_keeping_assistant/angle",
    #     Int8,
    #     obstacle_avoiding_cd,
    # )
    engine.spin()


main()
