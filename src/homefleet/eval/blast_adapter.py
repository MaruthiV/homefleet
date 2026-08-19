import subprocess
import tempfile
import numpy as np
from homefleet.paths import ROOT

BLAST_PY = ROOT / "envs" / "blast" / "bin" / "python"


def blast_fade(res, params, soh, temp_c_15min, dt_hours):
    # soc trace -> lfp fade per home via the blast venv; temp = ambient proxy (cabinet model todo)
    e = res["e"]
    E = params.energy_kwh * soh
    soc = np.clip(e / E, 0, 1)
    with tempfile.TemporaryDirectory() as td:
        inp, out = f"{td}/in.npz", f"{td}/out.npz"
        np.savez(inp, soc=soc.astype(np.float32), temp_c=np.asarray(temp_c_15min, dtype=np.float32), dt_s=dt_hours * 3600.0)
        subprocess.run([str(BLAST_PY), str(ROOT / "scripts" / "blast_worker.py"), inp, out], check=True, capture_output=True)
        return np.load(out)["fade"]
