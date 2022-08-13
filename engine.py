#! /usr/bin/env python3

from time import sleep
from deps import get_map_handler, get_plan_nodes
from node import Node
from std_msgs.msg import String, Int8
from settings import settings
from std_msgs.msg import Bool


map_handler = get_map_handler()
plan = get_plan_nodes()


obstacle_detector_subscriber = "obstacle_detector_subscriber"
lane_keeping_assistant_subscriber = "lane_keeping_assistant_subscriber"
localization_subscriber = "localization_subscriber"
traffic_light_subscriber = "traffic_light_subscriber"
road_sign_subscriber = "road_sign_subscriber"

commands_publisher = "commands"

engine = Node("engine_node")

status = {"moving": False, "obstacle": False, "angle": 0.0, "current_plan_node": 0, "traffic_red": False, "stop_sign": False}


def make_decision():
    global status

    if status["moving"]:
        if status["obstacle"] or status["stop_sign"] or status["traffic_red"]:
            engine.publish(commands_publisher, settings.STOP_ENGINE_COMMAND)
            status["moving"] = False
            print("STOP ENGINE")
        
        # else:
            # if status["angle"] >= 1:
            #     engine.publish(commands_publisher, settings.LEFT_ENGINE_COMMAND)
            #     status["angle"] -= 1
            #     print("GO LEFT")
                # sleep(0.2)
                # engine.publish(commands_publisher, settings.RIGHT_ENGINE_COMMAND)
            # elif status["angle"] <= -1:
            #     engine.publish(commands_publisher, settings.RIGHT_ENGINE_COMMAND)
            #     status["angle"] += 1
            #     print("GO RIGHT")
                # sleep(0.2)
                # engine.publish(commands_publisher, settings.LEFT_ENGINE_COMMAND)
    else:
        if not status["obstacle"] and  not status["stop_sign"] and not status["traffic_red"]:
            engine.publish(commands_publisher, settings.GO_ENGINE_COMMAND)
            status["moving"] = True
            print("START")

    status["stop_sign"] = False
def obstacle_avoiding_cb(msg: Int8):
    global status

    rangeCode = msg.data

    if rangeCode == settings.CLOSE_OBSTACLE_CODE:
        status["obstacle"] = True
    else:
        status["obstacle"] = False

    make_decision()


def lane_keeping_assistant_cb(msg: Int8):
    global status

    value = msg.data

    if status["moving"]:
        if value > 0:
            status["angle"] += float(value / 100)
        elif value < 0:
            status["angle"] -= float(value / 100)

    make_decision()

    # engine.publish(commands_publisher, settings.LEFT_ENGINE_COMMAND)




def localization_cb(msg: Bool):
    is_left = msg.data
    if is_left == True:
        engine.publish(commands_publisher, settings.STOP_ENGINE_COMMAND)
        status["current_plan_node"] += 1
        print(f"I am in {plan[status['current_plan_node']]}")
        map_handler.move(plan[status["current_plan_node"]])
        current_node = map_handler.get_currnet_node()
        sleep(3)
        engine.publish(commands_publisher, settings.STEER_ENGINE_COMMAND.format(current_node.steering_angle))
        sleep(2)
        if status["moving"] == True:
            engine.publish(commands_publisher, settings.GO_ENGINE_COMMAND)


def traffic_light_cb(msg: Bool):
    traffic_value = msg.data
    if traffic_value == False:
        # engine.publish(commands_publisher, settings.STOP_ENGINE_COMMAND)
        status["traffic_red"] = True
    else: 
        status["traffic_red"] = False
    make_decision()
    



def road_sign_cb(msg: String):
    sign_value = msg.data
    if sign_value == "Stop":
        status["stop_sign"] = True
    else:
        status["stop_sign"] = False
    make_decision()


def main():
    engine.init_publisher(commands_publisher, "engine/commands", String)

    engine.init_subscriber(
        obstacle_detector_subscriber, "/algo/obstacle", Int8, obstacle_avoiding_cb
    )
    engine.init_subscriber(
        lane_keeping_assistant_subscriber,
        "algo/lane_keeping_assistant",
        Int8,
        lane_keeping_assistant_cb,
    )
    engine.init_subscriber(localization_subscriber, "/algo/localization", Bool, localization_cb)
    engine.init_subscriber(traffic_light_subscriber, "algo/traffic", Bool, traffic_light_cb)
    engine.init_subscriber(road_sign_subscriber, "algo/sign", String, road_sign_cb)

    # sleep(5)
    engine.publish(commands_publisher, settings.GO_ENGINE_COMMAND)
    status['moving'] = True
    engine.spin()


main()
