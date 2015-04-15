import os

import doit
from doit.tools import check_timestamp_unchanged


def task_compliance():
    def run_tests(backend):
        action = 'py.test --pyargs {0} > results/{0}.txt'.format(backend)
        directory = os.path.join(os.pardir, backend)
        return {'name': backend,
                'actions': [action],
                'targets': ['results/{0}.txt'.format(backend)],
                'uptodate': [check_timestamp_unchanged(directory)]}
    yield run_tests('nengo')
    yield run_tests('nengo_reference')

if __name__ == '__main__':
    doit.run(locals())
