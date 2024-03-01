from enum import Enum


class TrainingStatus(Enum):
    SUCCESS = 0
    FAILED = -1
    NOT_STARTED = 1

    PREPARING = 2
    TRAINING = 3
