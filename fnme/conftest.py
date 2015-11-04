"""py.test stuff that runs the benchmarks."""

import importlib
import tempfile

import pytest

import nengo
from nengo.tests.conftest import *
from nengo.utils.stdlib import Timer


if nengo.__version__ == '2.0.1':
    # Monkey patch hacks to solve some problems in Nengo 2.0.1
    def pl_init(self, simulator, module, function, nl=None, plot=None):
        if plot is None:
            self.plot = int(os.getenv("NENGO_TEST_PLOT", 0))
        else:
            self.plot = plot

        self.dirname = "%s.plots" % simulator.__module__
        if nl is not None:
            self.dirname = os.path.join(self.dirname, nl.__name__)

        modparts = module.__name__.split('.')[1:]
        self.filename = "%s.%s.pdf" % ('.'.join(modparts), function.__name__)

    nengo.utils.testing.Plotter.__init__ = pl_init
else:
    # Monkey patch hacks to solve some problems in Nengo master
    def get_filename(self, ext=''):
        modparts = self.module_name.split('.')[1:]
        return "%s.%s.%s" % ('.'.join(modparts), self.function_name, ext)
    nengo.utils.testing.Recorder.get_filename = get_filename


def add_speed_profiling(Simulator):
    def patched_init(old_f):
        def __init__(self, *args, **kwargs):
            with Timer() as t:
                old_f(self, *args, **kwargs)
            Simulator.build_time = t.duration
        return __init__

    def patched_run(old_f):
        def run(self, *args, **kwargs):
            with Timer() as t:
                old_f(self, *args, **kwargs)
            Simulator.run_time = t.duration
        return run

    # Monkey patch in timing info for `__init__` (build) and `run`
    Simulator.__init__ = patched_init(Simulator.__init__)
    Simulator.run = patched_run(Simulator.run)

_Simulator = None
outdir = None
_profile = True


def pytest_configure(config):
    global _Simulator
    if not config.getoption('simulator'):
        raise ValueError("Please provide a simulator to benchmark.")
    _Simulator = load_class(config.getoption('simulator')[0])

    global profile
    profile = config.getoption('noprofile')
    if profile:
        add_speed_profiling(_Simulator)

    if config.getoption('seed'):
        # Change up the seed to get new networks
        nengo.tests.conftest.test_seed = config.getoption('seed')[0]

    global outdir
    outdir = "results/probes/%s" % _Simulator.__module__

    if not os.path.exists(outdir):
        os.makedirs(outdir)


def load_class(fully_qualified_name):
    mod_name, cls_name = fully_qualified_name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


@pytest.fixture
def outfile(request, seed):
    path = os.path.join(outdir, "%s-%s.txt" % (
        request.function.__name__, nengo.tests.conftest.test_seed))
    if profile:
        outf = open(path, 'w')
        outf.write('{\n')
    else:
        outf = tempfile.TemporaryFile()
    return outf


@pytest.fixture
def Simulator(request, outfile):
    def save_profiling():
        outfile.write('"buildtime": %f,\n' % _Simulator.build_time)
        outfile.write('"runtime": %f\n}\n' % _Simulator.run_time)
        del _Simulator.build_time
        del _Simulator.run_time
    request.addfinalizer(outfile.close)
    if profile:
        request.addfinalizer(save_profiling)
    return _Simulator
