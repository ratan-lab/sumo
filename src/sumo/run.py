""""Provides entry point main()."""

from sumo.command_line import parse_args
from sumo.modes import SUMO_MODES
import sys


def main():
    args = parse_args(sys.argv[1:])
    if args.command:
        mode = SUMO_MODES[args.command](**vars(args))
        mode.run()
