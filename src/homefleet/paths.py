from pathlib import Path
import os

_source_root = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('HOMEFLEET_ROOT', _source_root if (_source_root / 'pyproject.toml').is_file() else Path.cwd())).resolve()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
CONFIGS = ROOT / 'configs' if (ROOT / 'configs' / 'base.yaml').is_file() else Path(__file__).parent / 'configs'
EXPERIMENTS = ROOT / "experiments"
