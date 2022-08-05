#! /usr/bin/env python3


from node import Node
from sensor_msgs.msg import Range

dist_node = Node("distance_viewer")


def view_dist(msg: Range):
    print(f"The front distance is: {msg.range}")


dist_node.init_subscriber(
    "distance_ultra", "/raspberry/data/distance", Range, view_dist
)

dist_node.spin()
