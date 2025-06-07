import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(ROOT_DIR, 'hace-kv-optimization')
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from experiments import run_experiment

if __name__ == '__main__':
    run_experiment.main()
