def pytest_addoption(parser):
    # Make simulator passable
    parser.addoption('--simulator', nargs=1, type=str, default=None,
                     help='Specify simulator under test.')
