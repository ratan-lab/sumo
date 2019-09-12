from .evaluate import *
from .prepare import *
from .run import *
from sumo.constants import SUMO_COMMANDS

SUMO_MODES = {"prepare": SumoPrepare,
              "run": SumoRun,
              "evaluate": SumoEvaluate}

assert all([mode in SUMO_COMMANDS for mode in SUMO_MODES.keys()])
