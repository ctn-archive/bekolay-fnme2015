from nengo.tests.options import pytest_addoption as _nengooptions

def pytest_addoption(parser):
    # Make simulator passable
    parser.addoption('--simulator', nargs=1, type=str, default=None,
                     help='Specify simulator under test.')

    _nengooptions(parser)
