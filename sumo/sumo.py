""""Provides entry point main()."""

from .command_line import parse_args
from .modes import SUMO_MODES
import sys


def main():
    args = parse_args(sys.argv[1:])
    if args.command:
        mode = SUMO_MODES[args.command](**vars(args))
        mode.run()
