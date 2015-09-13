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

    def forsurecompile(fname, bibtex=True):
        pdf = CmdAction('pdflatex -interaction=nonstopmode %s.tex' % fname,
                        cwd=d)
        bib = CmdAction('bibtex %s' % fname, cwd=d)
        pdf_file = os.path.join(d, '%s.pdf' % fname)
        tex_file = os.path.join(d, '%s.tex' % fname)
        bib_file = os.path.join(d, '%s.bib' % fname)
        return {'name': fname,
                'file_dep': [tex_file, bib_file] if bibtex else [tex_file],
                'actions': [pdf, bib, pdf, pdf] if bibtex else [pdf, pdf],
                'targets': [pdf_file]}
    yield forsurecompile('paper')


def task_compliance():
    def run_tests(backend, pytest_args, cwd):
        result = '{}/results/{}.txt'.format(root, backend)
        pytest = ('py.test --neurons nengo.Direct,nengo.LIF --slow --plots '
                  '--analytics -- {} > {}'.format(pytest_args, result))
        action = CmdAction(pytest, cwd=cwd)
        return {'name': backend,
                'actions': ['rm -f {}'.format(result), action],
                'targets': [result],
                'uptodate': [check_timestamp_unchanged(cwd)]}
    yield run_tests('nengo_ocl', 'nengo_ocl/tests/test_sim_ocl.py',
                    os.path.join(os.pardir, 'nengo_ocl'))
    yield run_tests('nengo_distilled', 'nengo_distilled/tests/test_nengo.py',
                    os.path.join(os.pardir, 'nengo_distilled'))
    yield run_tests('nengo_brainstorm', 'nengo_brainstorm/tests/test_nengo.py',
                    os.path.join(os.pardir, 'nef-chip-hardware'))


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


def task_figures():
    yield {'name': 'models',
           'actions': [CmdAction(
               'py.test -p fnme.options --simulator nengo.Simulator '
               '--seed 5 --plot plots --noprofile -- fnme/benchmarks.py',
               cwd=root)]}
    yield {'name': 'accuracy',
           'actions': [(fnme.plots.accuracy,)]}
    yield {'name': 'speed',
           'actions': [(fnme.plots.speed, (True,))]}
    yield {'name': 'speed-np',
           'actions': [(fnme.plots.speed, (False,))]}
    yield {'name': 'combine',
           'actions': [(fnme.plots.fig1,),
                       (fnme.plots.fig2,),
                       (fnme.plots.fig3,),
                       (fnme.plots.fig4,)]}


if __name__ == '__main__':
    doit.run(locals())
