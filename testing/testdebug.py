




import matplotlib
import sys
from pathlib import Path

print("matplotlib imported from:", matplotlib.__file__)
print("matplotlib rcParams:", matplotlib.rcParams)

# Search for matplotlib.py or matplotlib folders in workspace
workspace = Path(__file__).parent
print("\nSearching for matplotlib.py and matplotlib folders in workspace:")
for path in workspace.rglob('matplotlib*'):
    print("Found:", path)

# Search sys.path for matplotlib.py or matplotlib folders
print("\nSearching sys.path for matplotlib.py and matplotlib folders:")
for p in sys.path:
    p_path = Path(p)
    if p_path.exists():
        for item in p_path.glob('matplotlib*'):
            print("Found in sys.path:", item)