from argparse import Namespace
from sumo import command_line
import pytest


def test_parse_args(capsys):
    correct = [
        "./sumo.py prepare infile type outfile",
        "./sumo.py prepare infile1,infile2 type outfile",
        "./sumo.py prepare infile type1,type2 outfile",
        "./sumo.py run infile 2 outfile",
        "./sumo.py"
    ]

    for cmd in correct:
        args = command_line.parse_args(cmd.split()[1:])
        assert isinstance(args, Namespace)

    incorrect = [
        "./sumo.py prepare",
        "./sump.py prepare infile"
        "./sumo.py prepare infile outfile",
        "./sumo.py run",
        "./sumo.py run infile",
        "./sumo.py run infile 2",
        "./sumo.py run infile outfile",
        "./sumo.py run infile two outfile",
        "./sumo.py run infile 2.5 outfile",
        "./sumo.py not_a_command"
    ]

    for cmd in incorrect:
        with pytest.raises(SystemExit):
            command_line.parse_args(cmd.split()[1:])
        _ = capsys.readouterr()  # catch stdout
