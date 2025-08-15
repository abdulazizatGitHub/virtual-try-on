from pathlib import Path

SRC: Path = Path(__file__).resolve().parent.parent

PROJECT: Path = SRC.parent

DATA: Path = PROJECT / 'data'
RESULTS: Path = DATA / 'results'
UPLOADS: Path = DATA / 'uploads'
LOGS: Path = DATA / 'logs'

def ensure_directories() -> None:
    for i in list(vars().values()):
        if isinstance(i, Path):
            i.mkdir(parents=True, exist_ok=True)