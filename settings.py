from dotenv import load_dotenv
import os

load_dotenv('.env')


class Settings():
    #commands
    STOP_COMMAND = "stop"
    GO_COMMAND = "go"
    LEFT_COMMAND = "left"
    RIGHT_COMMAND = "right"

    OBSTACLE_RANGE = os.getenv('OBSTACLE_RANGE')
    











settings = Settings()