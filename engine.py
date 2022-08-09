#! /usr/bin/env python3

from node import Node
from std_msgs.msg import String, Int8
from settings import settings

engine_obstacle_detector_subscriber = "obstacle_detector_subscriber"
engine = Node("engine_node")


def obstacle_cd(msg: Int8):
    range = msg.data
    command = ""
    # if range == settings.CLOSE_OBSTACLE:
    #     command = settings.STOP_ENGINE_COMMAND
    # else :
    #     command = settings.GO_ENGINE_COMMAND
    if range >= 0:
        command = settings.LEFT_ENGINE_COMMAND
    else:
        command = settings.RIGHT_ENGINE_COMMAND
    engine.publish("commands", command)
    print(f"angle: {range}")


def main():
    engine.init_publisher("commands", "engine/commands", String)
    engine.init_subscriber(
        engine_obstacle_detector_subscriber,
        "lane_keeping_assistant/angle",
        Int8,
        obstacle_cd,
    )
    engine.spin()


main()
