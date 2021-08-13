import shutil

try:
    shutil.rmtree('.\\.eggs')
except:
    pass

try:
    shutil.rmtree('.\\.pytest_cache')
except:
    pass

try:
    shutil.rmtree('.\\qdms.egg-info')
except:
    pass

try:
    shutil.rmtree('.\\qdms\\__pycache__')
except:
    pass

try:
    shutil.rmtree('.\\tests\\__pycache__')
except:
    pass

try:
    shutil.rmtree('.\\tests\\Simulation')
except:
    pass

try:
    shutil.rmtree('.\\Simulation')
except:
    pass
