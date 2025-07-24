

"""
Monkey Patching 

This is for fixing bugs (or improve features).

For each external lib there are: 
    1. bug_patcher.py
        For all fixing bug monkey patching (like curriculumagent/bug_patcher.py.py) 
        that contains the monkey patching to fix bug (they can be import altogether) 
    2. improvements_in_general.py 
        For all improvements in general (like curriculumagent/improvements_in_general.py) 
        that contains the monkey patching to improve features that don't 
        (or might not) have conflicts, so they can be import altogether 
    3. each_feature.py
        For each intensive adding/modification of features (that should not 
        be used in an arbitrary manner cause they can cause conflicts, 
        should be used with care)
        (like curriculumagent/not_use_dropout_anymore.py) 
    

Usage: 
    If you want to monkey patch something, import the corresponding file, like:
    >>import MyOwnLib.MonkeyPatch.curriculumagent.monkey_patcher
    If you want to monkey patch the whole lib, just import the corresponding lib folder, like:
    >>import MyOwnLib.MonkeyPatch.curriculumagent
    If you want to monkey patch everthing:
    >>import MyOwnLib.MonkeyPatch
    or 
    >>import MyOwnLib
  
        
"""



#from . import curriculumagent


# Automatically import everything in the package, files and subpackages.

""" 

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



"""



            




