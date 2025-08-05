import os
from pathlib import Path

def DATA_DIR(dataset_name):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base_dir, 'datasets', dataset_name)


