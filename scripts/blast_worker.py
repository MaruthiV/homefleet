import sys
import numpy as np

# runs inside envs/blast (numpy<2); reads npz path with soc (T,N), temp_c (T,), dt_s; writes fade npz
def main():
    inp, out = sys.argv[1], sys.argv[2]
    from blast import models
    z = np.load(inp)
    soc = z["soc"]
    temp = z["temp_c"]
    dt_s = float(z["dt_s"])
    T, N = soc.shape
    t = np.arange(T) * dt_s
    fade = np.zeros(N)
    for j in range(N):
        cell = models.Lfp_Gr_SonyMurata3Ah_Battery()
        cell.simulate_battery_life({"Time_s": t, "SOC": soc[:, j].astype(float), "Temperature_C": temp.astype(float)})
        fade[j] = 1.0 - float(cell.outputs["q"][-1])
    np.savez(out, fade=fade)


if __name__ == "__main__":
    main()
