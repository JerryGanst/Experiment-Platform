# H2O实验包

# No sys.path modification here anymore, as it didn't work as expected
# for top-level imports in the main execution script when using `python -m`.
# The PYTHONPATH environment variable will be used instead for broader visibility.

# import os # No longer needed here

# 获取 __init__.py 所在的目录 (即 h2o_experiment 目录)
# package_dir = os.path.dirname(os.path.abspath(__file__)) # No longer needed here
# 获取项目根目录 (即 h2o_experiment 目录的父目录)
# project_root_dir = os.path.dirname(package_dir) # No longer needed here

# print(f"DEBUG from h2o_experiment/__init__.py: project_root_dir = {project_root_dir}")
# print(f"DEBUG from h2o_experiment/__init__.py: sys.path AFTER insert = {sys.path}") 