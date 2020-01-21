from .evaluate import *
from .interpret import *
from .prepare import *
from .run import *
from sumo.constants import SUMO_COMMANDS

SUMO_MODES = {"prepare": SumoPrepare,
              "run": SumoRun,
              "evaluate": SumoEvaluate,
              "interpret": SumoInterpret}

assert all([mode in SUMO_COMMANDS for mode in SUMO_MODES.keys()])
