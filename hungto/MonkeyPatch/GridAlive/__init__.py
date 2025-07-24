




# Automatically import everything in the package, files and subpackages.

import os
import importlib

# Get the directory of this __init__.py file

import os
import importlib

# Get the directory of this __init__.py file
package_dir = os.path.dirname(__file__)
package_name = __name__

for filename in os.listdir(package_dir):
    # Import all .py files except __init__.py and setup.py
    if filename in ("__init__.py", "setup.py"):
        continue
    if filename.endswith(".py"):
        # Import the module
        module_name = filename[:-3]
        mod = importlib.import_module(f"{package_name}.{module_name}")
        globals()[module_name] = mod  # <-- Add to namespace

    # Import all subfolders that contain an __init__.py (i.e., subpackages)
    elif os.path.isdir(os.path.join(package_dir, filename)):
        if os.path.exists(os.path.join(package_dir, filename, "__init__.py")):
            module_name = filename
            mod = importlib.import_module(f"{package_name}.{filename}")
            globals()[module_name] = mod  # <-- Add to namespace









