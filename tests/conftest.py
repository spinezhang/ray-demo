import os
import pytest

@pytest.fixture(scope="function")
def work_path():
    cur_path = os.getcwd()
    if os.path.basename(cur_path) == 'tests':
        return cur_path + '/..'
    else:
        return cur_path
