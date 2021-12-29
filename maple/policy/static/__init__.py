import sys
import os
import importlib
import pdb


def import_fns(path, file, fns_name='StaticFns'):
	full_path = os.path.join("maple/policy/static", file)
	import_path = full_path.replace('/', '.')
	module = importlib.import_module(import_path)
	fns = getattr(module, fns_name)
	return fns
def get_base_path():
	return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cwd = os.path.join(get_base_path(), 'policy/static')
files = os.listdir(cwd)
## remove __init__.py
files = filter(lambda x: '__' not in x and x[0] != '.', files)
## env.py --> env
files = map(lambda x: x.replace('.py', ''), files)

## {env: StaticFns, ... }
static_fns = {file: import_fns(cwd, file) for file in files}

sys.modules[__name__] = static_fns

