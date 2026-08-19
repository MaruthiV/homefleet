from dataclasses import dataclass
import numpy as np
import yaml
from homefleet.paths import CONFIGS


@dataclass
class BatteryParams:
    energy_kwh: np.ndarray
    p_charge_kw: np.ndarray
    p_discharge_kw: np.ndarray
    eta_c: float
    eta_d: float
    soc_min: float


def load_presets():
    with open(CONFIGS / "battery_presets.yaml") as f:
        return yaml.safe_load(f)


def make_fleet(preset_names, presets=None):
    presets = presets or load_presets()
    com = presets["common"]
    e = np.array([presets[n]["energy_kwh"] for n in preset_names], dtype=float)
    pc = np.array([presets[n]["p_charge_kw"] for n in preset_names], dtype=float)
    pd_ = np.array([presets[n]["p_discharge_kw"] for n in preset_names], dtype=float)
    return BatteryParams(e, pc, pd_, com["eta_charge"], com["eta_discharge"], com["soc_min_physical"])


def clip_power(e, pc, pd_, params, soh, avail, floor, dt):
    # clip requested charge/discharge to inverter, availability, and energy bounds
    pc = np.clip(pc, 0, params.p_charge_kw * avail)
    pd_ = np.clip(pd_, 0, params.p_discharge_kw * avail)
    emax = params.energy_kwh * soh
    room = np.maximum(emax - e, 0) / (dt * params.eta_c)
    spare = np.maximum(e - floor, 0) * params.eta_d / dt
    return np.minimum(pc, room), np.minimum(pd_, spare)


def step_energy(e, pc, pd_, params, dt):
    return e + dt * (params.eta_c * pc - pd_ / params.eta_d)


def envelope(e, params, soh, avail, floor, dt):
    # max aggregate charge / discharge power realizable this step
    emax = params.energy_kwh * soh
    pc_max = np.minimum(params.p_charge_kw * avail, np.maximum(emax - e, 0) / (dt * params.eta_c))
    pd_max = np.minimum(params.p_discharge_kw * avail, np.maximum(e - floor, 0) * params.eta_d / dt)
    return pc_max, pd_max
