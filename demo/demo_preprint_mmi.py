#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo reproducible del preprint: simulación MMI y calibración pH-λ.
Genera:
 - figures/figura_espectros_MMI.png
 - figures/calibracion_pH_lambda.png
 - demo/calibration_points.csv
 - demo/montecarlo_summary.csv

Requisitos: numpy, scipy, matplotlib
Ejecuta desde la carpeta del repo:
    cd demo
    python demo_preprint_mmi.py
"""
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import csv

# ------------------ Parámetros del modelo óptico ------------------
L = 50e-3          # longitud efectiva (m)
n0 = 1.45          # índice base
alpha = 1.2e-3     # sensibilidad dn/dpH
m = np.arange(1, 201)  # órdenes de interferencia
pH_values = np.linspace(2.5, 8.0, 10)  # 10 puntos de pH
noise_sigma_nm = 0.02  # ruido (nm) en la intensidad simulada (escala arbitraria)

# ------------------ Salidas ------------------
repo_root = Path(__file__).resolve().parents[1]
fig_dir = repo_root / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

demo_dir = Path(__file__).resolve().parent
calib_csv = demo_dir / "calibration_points.csv"
mc_csv = demo_dir / "montecarlo_summary.csv"

# ------------------ Funciones ------------------
def simulate_spectrum(pH, L=L, n0=n0, alpha=alpha, m=m, noise_sigma_nm=noise_sigma_nm):
    """
    Simula un espectro interferométrico MMI en nm (eje x) con una
    intensidad adimensional (eje y).
    """
    neff = n0 + alpha * (pH - 7.0)
    wl_nm = (2 * neff * L / m) * 1e9  # longitud de onda en nm
    wl_nm = np.sort(wl_nm)            # orden creciente

    # Señal interferométrica simplificada
    I = 1.0 + np.cos(2 * np.pi * neff * L / (wl_nm * 1e-9))
    I += np.random.normal(0.0, noise_sigma_nm, size=I.size)

    # Suavizado para emular preprocesamiento espectral
    I_smooth = savgol_filter(I, window_length=15, polyorder=3, mode="interp")
    return wl_nm, I_smooth

def find_peak_wavelength(wl_nm, I):
    """Devuelve la longitud de onda (nm) del máximo interferométrico."""
    idx = np.argmax(I)
    return float(wl_nm[idx])

def linear_fit(x, y):
    """Ajuste lineal sencillo usando polyfit. Regresa pendiente, intercepto, R^2."""
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2, y_hat

# ------------------ Simulación base: espectros ------------------
peak_points = []  # (pH, lambda_peak_nm)

plt.figure(figsize=(10, 5.5))
for pH in pH_values:
    wl_nm, I = simulate_spectrum(pH)
    plt.plot(wl_nm, I, linewidth=1.0, label=f"pH = {pH:.1f}")
    peak = find_peak_wavelength(wl_nm, I)
    peak_points.append((pH, peak))

plt.title("Respuesta interferométrica simulada (MMI) para distintos pH")
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Intensidad normalizada (u.a.)")
plt.legend(loc="best", ncol=2, fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(fig_dir / "figura_espectros_MMI.png", dpi=300)
plt.close()

# ------------------ Calibración pH - λ pico ------------------
points = np.array(peak_points)  # Nx2
pH_arr = points[:, 0]
lam_arr = points[:, 1]

slope, intercept, r2, lam_hat = linear_fit(pH_arr, lam_arr)

plt.figure(figsize=(7, 5))
plt.scatter(pH_arr, lam_arr, s=40, label="Datos simulados")
plt.plot(pH_arr, lam_hat, linewidth=2.0, label=f"Ajuste lineal (R² = {r2:.3f})")
plt.xlabel("pH")
plt.ylabel("λ pico (nm)")
plt.title("Calibración pH vs. longitud de onda pico")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "calibracion_pH_lambda.png", dpi=300)
plt.close()

# Guardar puntos de calibración
with calib_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["pH", "lambda_peak_nm"])
    writer.writerows(peak_points)

# ------------------ Mini Monte Carlo de sensibilidad ------------------
# Variar ligeramente alpha y ruido para ver sensibilidad en la pendiente
rng = np.random.default_rng(42)
mc_runs = 50
slopes = []
r2_list = []

for _ in range(mc_runs):
    alpha_mc = alpha * (1.0 + rng.normal(0, 0.05))  # ±5%
    noise_mc = max(0.005, noise_sigma_nm * (1.0 + rng.normal(0, 0.2)))  # >= 0.005

    tmp_points = []
    for pH in pH_values:
        wl_nm, I = simulate_spectrum(pH, alpha=alpha_mc, noise_sigma_nm=noise_mc)
        tmp_points.append((pH, find_peak_wavelength(wl_nm, I)))
    tmp_points = np.array(tmp_points)
    s, b, r2_mc, _ = linear_fit(tmp_points[:, 0], tmp_points[:, 1])
    slopes.append(s)
    r2_list.append(r2_mc)

# Guardar resumen MC
with mc_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "slope_nm_per_pH", "r2"])
    for i, (s, r) in enumerate(zip(slopes, r2_list), 1):
        writer.writerow([i, s, r])

print("Listo ✅")
print("Figuras guardadas en:", fig_dir.as_posix())
print("CSV de calibración:", calib_csv.as_posix())
print("CSV Monte Carlo:", mc_csv.as_posix())
print(f"Sensibilidad estimada (pendiente): {slope:.3f} nm/pH | R² = {r2:.3f}")
