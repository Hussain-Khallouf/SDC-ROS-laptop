class Settings:
    # Variables
    OBSTACLE_CLOSE_RANGE = 10
    OBSTACLE_MIDDLE_RANGE = 20

    # Engine commands
    STOP_ENGINE_COMMAND = "stop"
    GO_ENGINE_COMMAND = "go"
    LEFT_ENGINE_COMMAND = "left"
    RIGHT_ENGINE_COMMAND = "right"

    # The output of obstacle detection algorithm
    FAR_OBSTACLE_CODE = 0
    MIDDLE_OBSTACLE_CODE = 1
    CLOSE_OBSTACLE_CODE = 2

    # Lane Keeping Assistant
    STRAIGHT = 0
    LEFT_ANGLE = 1
    RIGHT_ANGLE = 2

    # The output of traffic recognition algorithm
    NO_TRAFFIC = 0
    GREEN_TRAFFIC = 1
    RED_TRAFFIC = 2

    # The output of road sign recognition algorithm
    STOP_SIGN = 0


    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 360

settings = Settings()
