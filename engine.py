#! /usr/bin/env python3

from time import sleep
from deps import get_map_handler
from node import Node
from std_msgs.msg import String, Int8
from settings import settings

map_handler = get_map_handler()

obstacle_detector_subscriber = "obstacle_detector_subscriber"
lane_keeping_assistant_subscriber = "lane_keeping_assistant_subscriber"
localization_subscriber = "localization_subscriber"
traffic_light_subscriber = "traffic_light_subscriber"
road_sign_subscriber = "road_sign_subscriber"

commands_publisher = "commands"

engine = Node("engine_node")

status = {'moving': False, 'obstacle': False, 'angle': settings.STRAIGHT}

def obstacle_avoiding_cb(msg: Int8):
    global status

    rangeCode = msg.data
    
    if rangeCode == settings.CLOSE_OBSTACLE_CODE:
        status['obstacle'] = True
    else:
        status['obstacle'] = False
    
    make_decision()

def lane_keeping_assistant_cb(msg: Int8):
    global status

    value = msg.data

    if status['moving']:
        if value > 0:
            status['angle'] = settings.LEFT_ANGLE
        else: 
            status['angle'] = settings.RIGHT_ANGLE
    
    make_decision()

        # engine.publish(commands_publisher, settings.LEFT_ENGINE_COMMAND)

def make_decision():
    global status

    if status['moving']:
        if status['obstacle']:
            engine.publish(commands_publisher, settings.STOP_ENGINE_COMMAND)
            status['moving'] = False
            print('STOP ENGINE')
        else:
            if status['angle'] == settings.LEFT_ANGLE:
                engine.publish(commands_publisher, settings.LEFT_ENGINE_COMMAND)
                status['angle'] = settings.STRAIGHT
                print('GO LEFT')
            elif status['angle'] == settings.RIGHT_ANGLE:
                engine.publish(commands_publisher, settings.RIGHT_ENGINE_COMMAND)
                status['angle'] = settings.STRAIGHT
                print('GO RIGHT')
    else:
        if not status['obstacle']:
            engine.publish(commands_publisher, settings.GO_ENGINE_COMMAND)
            status['moving'] = True
            print('START')


def localization_cb(msg):
    pass

def traffic_light_cb(msg: Int8):
    traffic_value = msg.data
    if traffic_value == settings.RED_TRAFFIC:
        engine.publish(commands_publisher, settings.STOP_ENGINE_COMMAND)
    

def road_sign_cb(msg: Int8):
    sign_value = msg.data
    if sign_value == settings.STOP_SIGN:
        engine.publish(commands_publisher, settings.STOP_ENGINE_COMMAND)

def main():
    engine.init_publisher(commands_publisher, "engine/commands", String)

    engine.init_subscriber(obstacle_detector_subscriber, "/algo/obstacle", Int8, obstacle_avoiding_cb)
    engine.init_subscriber(lane_keeping_assistant_subscriber, "algo/lane_keeping_assistant", Int8, lane_keeping_assistant_cb)
    # engine.init_subscriber(localization_subscriber, "", Int8, localization_cb)
    # engine.init_subscriber(traffic_light_subscriber, "", Int8, traffic_light_cb)
    # engine.init_subscriber(road_sign_subscriber, "", Int8, road_sign_cb)

    sleep(5)
    engine.publish(commands_publisher, settings.GO_ENGINE_COMMAND)
    engine.spin()

main()
