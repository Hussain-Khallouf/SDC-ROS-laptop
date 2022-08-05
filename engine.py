#! /usr/bin/env python3

from node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import String
from settings import settings

engine = Node("engine_node")

engine.init_publisher("commands", "engine/commands", String)


def dist_cb(msg: Range):
    range = int(msg.range)
    if range < int(settings.OBSTACLE_RANGE):
        engine.publish("commands", settings.STOP_COMMAND)
    else:
        engine.publish("commands", settings.GO_COMMAND)


engine.init_subscriber("distance", "raspberry/data/distance", Range, dist_cb)

engine.spin()