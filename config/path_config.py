from pathlib import Path

# Get the project root directory (where this config file is located)
PROJECT: Path = Path(__file__).resolve().parent.parent

# Data directories relative to project root
DATA: Path = PROJECT / 'data'
RESULTS: Path = DATA / 'results'
UPLOADS: Path = DATA / 'uploads'
LOGS: Path = DATA / 'logs'

# Ensure directories exist on module import
for i in list(vars().values()):
        if isinstance(i, Path):
            i.mkdir(parents=True, exist_ok=True)