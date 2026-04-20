import sys
import os

# Adding the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Adding the src directory to sys.path so Python can find all modules
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
