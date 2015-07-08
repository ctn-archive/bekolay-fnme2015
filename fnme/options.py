from nengo.tests.options import pytest_addoption as _nengooptions


def pytest_addoption(parser):
    # Make simulator passable
    parser.addoption('--simulator', nargs=1, type=str, default=None,
                     help='Specify simulator under test.')
    parser.addoption('--seed', nargs=1, type=int, default=None,
                     help='Specify the global test seed.')
    parser.addoption('--noprobes', action='store_false', default=True,
                     help='Disable probing.')
    parser.addoption('--noprofile', action='store_false', default=True,
                     help='Disable speed profiling.')

    _nengooptions(parser)
