import json

from map_handler import MapHandler, MapNode
from typing import List


def get_plan_nodes() -> List:
    with open("/home/hussain/project2_ws/src/self_driving_car_01/scripts/plan.txt", "r") as plan_file:
        plan_list = plan_file.read().split(",")
    return plan_list


def get_map_nodes(file_path: str) -> List[MapNode]:
    with open(file_path, "r") as map_file:
        map = json.load(map_file)
        map_nodes = [MapNode.parse_obj(node) for node in map["nodes"]]
    return map_nodes


def get_map_handler():
    return MapHandler(get_map_nodes("/home/hussain/project2_ws/src/self_driving_car_01/scripts/map.json"), "start")
