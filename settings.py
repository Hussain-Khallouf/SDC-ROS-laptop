


class Settings():
    #Variables
    OBSTACLE_CLOSE_RANGE = 20
    OBSTACLE_MIDDLE_RANGE = 50
    
    #Engine commands
    STOP_ENGINE_COMMAND = "stop"
    GO_ENGINE_COMMAND = "go"
    LEFT_ENGINE_COMMAND = "left"
    RIGHT_ENGINE_COMMAND = "right"

    #The output of obstacle detection algorithm
    FAR_OBSTACLE = 0
    MIDDLE_OBSTACLE = 1
    CLOSE_OBSTACLE = 2

    #The output of traffic recognition algorithm
    NO_TRAFFIC = 0
    GREEN_TRAFFIC = 1
    RED_TRAFFIC = 2





settings = Settings()