#! /usr/bin/env python3


from node import Node
from sensor_msgs.msg import Range

def view_dist(msg: Range):
        print(f"The front distance is: {msg.range}")
        
def main():
    dist_node = Node("distance_viewer")

    dist_node.init_subscriber(
        "distance_ultra", "/raspberry/data/distance", Range, view_dist
    )

    dist_node.spin()



main()