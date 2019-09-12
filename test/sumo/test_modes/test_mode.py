from sumo.modes import mode
import pytest


def test_run():
    class SomeMode(mode.SumoMode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def run(self):
            super(SomeMode, self).run()

    for obj in [SomeMode(), SomeMode(some_arg="value")]:
        with pytest.raises(NotImplementedError):
            obj.run()
