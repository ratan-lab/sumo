""" Setup script for the sumo application.

"""
import re
from os import walk
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

_config = {
    "name": "python-sumo",
    "url": "https://github.com/ratan-lab/sumo",
    "author": "Karolina Sienkiewicz",
    "author_email": "sienkiewicz2k@gmail.com",
    "package_dir": {"": "sumo"},
    "packages": find_packages("sumo"),

    "entry_points": {
        "console_scripts": ("sumo = sumo.sumo:main",),
    },
    "classifiers": [
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    "data_files": ("etc/",),
}


def main() -> int:
    """ Execute the setup command.

    """

    def data_files(*paths):
        """ Expand path contents for the `data_files` config variable.  """
        for path in map(Path, paths):
            if path.is_dir():
                for root, _, files in walk(str(path)):
                    yield root, tuple(str(Path(root, name)) for name in files)
            else:
                yield str(path.parent), (str(path),)
        return

    def version():
        """ Get the local package version. """
        return re.search('^__version__\s*=\s*"(.*)"', open('sumo/constants.py').read(), re.M).group(1)

    _config.update({
        "data_files": list(data_files(*_config["data_files"])),
        "version": version(),
    })
    setup(**_config)
    return 0


# Make the script executable.

if __name__ == "__main__":
    raise SystemExit(main())
