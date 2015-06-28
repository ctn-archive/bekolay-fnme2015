#!/usr/bin/env python

"""
Before running this, make sure the `fnme` module is installed,
by running `python setup.py develop`.
"""

import pkgutil
import os
import random

import doit
from doit.action import CmdAction
from doit.tools import check_timestamp_unchanged

import fnme.plots

root = os.path.dirname(os.path.realpath(__file__))


def task_reset():
    return {'actions': ['rm -rf ./results/probes',
                        'rm -rf ./results/noprobes']}


def task_paper():
    d = os.path.join(root, 'paper')
    yield {'name': 'main',
           'actions': [CmdAction('rubber --pdf paper.tex', cwd=d)],
           'targets': [os.path.join(d, 'paper.pdf')]}
    yield {'name': 'supplementary',
           'actions': [CmdAction('rubber --pdf supplementary.tex', cwd=d)],
           'targets': [os.path.join(d, 'supplementary.pdf')]}


def task_compliance():
    def run_tests(backend, pytest_args='', cwd='.'):
        result = '{}/results/{}.txt'.format(root, backend)
        pytest = ('py.test --benchmarks --optional'
                  ' {} > {}'.format(pytest_args, result))
        action = CmdAction(pytest, cwd=cwd)
        return {'name': backend,
                'actions': ['rm -f {}'.format(result), action],
                'targets': [result],
                'uptodate': [check_timestamp_unchanged(cwd)]}
    yield run_tests('nengo_ocl', 'nengo_ocl/tests/test_sim_ocl.py',
                    cwd=os.path.join(os.pardir, 'nengo_ocl'))
    yield run_tests('nengo_distilled', 'nengo_distilled/tests/test_nengo.py',
                    cwd=os.path.join(os.pardir, 'nengo_distilled'))
    yield run_tests('nengo_brainstorm', 'nengo_brainstorm/tests/test_nengo.py',
                    cwd=os.path.join(os.pardir, 'nef-chip-hardware'))


def task_benchmarks():
    seed = random.randint(10000, 99999)
    sims = [sim for sim in ('nengo',
                            'nengo_ocl',
                            'nengo_distilled',
                            'nengo_brainstorm',
                            'nengo_spinnaker')
            if pkgutil.find_loader(sim)]

    acts = ['py.test -p fnme.options --simulator {}.Simulator --seed '
            '{:d} %(pytestargs)s -- fnme/benchmarks.py'.format(sim, seed)
            for sim in sims]

    return {'actions': acts,
            'params': [{'name': 'pytestargs',
                        'short': 'a',
                        'default': '',
                        'help': 'Additional flags to pass onto py.test'}]}


def task_plots():
    yield {'name': 'accuracy',
           'actions': [(fnme.plots.accuracy,)]}
    yield {'name': 'speed',
           'actions': [(fnme.plots.speed, (True,))]}
    yield {'name': 'speed-np',
           'actions': [(fnme.plots.speed, (False,))]}


def task_combine():
    yield {'name': 'fig1',
           'actions': [(fnme.plots.fig1,)]}
    yield {'name': 'fig2',
           'actions': [(fnme.plots.fig2,)]}
    yield {'name': 'fig3',
           'actions': [(fnme.plots.fig3,)]}
    yield {'name': 'fig4',
           'actions': [(fnme.plots.fig4,)]}


if __name__ == '__main__':
    doit.run(locals())
