#!/usr/bin/env python

"""
Before running this, make sure the `fnme` module is installed,
by running `python setup.py develop`.
"""

import os

import doit
from doit.action import CmdAction
from doit.tools import check_timestamp_unchanged

import fnme.plots

root = os.path.dirname(os.path.realpath(__file__))


def task_reset():
    return {'actions': ['rm -rf ./results/nengo*']}


def task_paper():
    d = os.path.join(root, 'paper')
    return {'actions': [CmdAction('rubber --pdf paper.tex', cwd=d)],
            'targets': [os.path.join(d, 'paper.pdf')]}


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
    def run_benchmarks(sim):
        pytest = ('py.test -p fnme.options --simulator {}.Simulator -- '
                  'fnme/benchmarks.py'.format(sim))
        return {'name': sim, 'actions': [pytest]}
    yield run_benchmarks('nengo')
    yield run_benchmarks('nengo_ocl')
    yield run_benchmarks('nengo_distilled')
    yield run_benchmarks('nengo_brainstorm')
    yield run_benchmarks('nengo_spinnaker')


def task_plots():
    yield {'name': 'compliance',
           'actions': [(fnme.plots.compliance,)]}
    yield {'name': 'accuracy',
           'actions': [(fnme.plots.accuracy,)]}
    yield {'name': 'speed',
           'actions': [(fnme.plots.speed,)]}


if __name__ == '__main__':
    doit.run(locals())
