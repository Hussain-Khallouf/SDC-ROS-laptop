from typing import List
from pydantic import BaseModel



class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]



class MapNode(BaseModel):
    name: str
    steering_angle: int
    lane_keeping_assistant: bool
    next_nodes: List

    def __str__(self):
        return f"name: {self.name}, steeringAngle: {self.steering_angle}, laneKeepingAssistant: {self.lane_keeping_assistant}"


class MapHandler(metaclass=SingletonMeta):
    def __init__(
        self,
        map_nodes: List[MapNode],
        start_node_name: str
    ):
        self.map = map_nodes
        self.node_counts = len(map_nodes)
        self.current_node = next((node for node in self.map if node.name == start_node_name))

    def get_currnet_node(self) -> MapNode:
        return self.current_node

    def get_next_nodes(self) -> List[str]:
        return self.current_node.next_nodes

    def move(self, next_node_name: str):
        if next_node_name not in self.current_node.next_nodes:
            raise ValueError("No such next node")
        next_node = next((node for node in self.map if node.name == next_node_name))
        self.current_node = next_node
        return next_node

    def __str__(self):
        for node in self.map:
            print(
                f'{node.name}: next nodes = {node.next_nodes}'
            )
