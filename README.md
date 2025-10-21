# Smart-Viticulture-DCC
# Smart Viticulture – DCC (Doctorate in Computer Science, ITESM)

**Author:** Edgar Eduardo Peña Sandoval  
**Proposed Advisor:** Dr. Luis Alberto Muñoz-Ubando  
**Institution:** Tecnológico de Monterrey, Campus Monterrey  
**Email:** e.pena.sandoval01@gmail.com  
**Program:** Doctorado en Ciencias Computacionales (DCC)  

---

## 🧠 Project Overview

This repository accompanies the doctoral proposal:

> **“Smart Viticulture: Integrating Fiber-Optic Interferometric Sensing and Machine Learning for Vineyard Optimization.”**

The project proposes a **hybrid optical–AI sensing system** for vineyard monitoring, integrating **multimode fiber interferometry (MMI)**, **machine learning**, and **computer vision** to enable precision agriculture.  
It aligns with DCC research lines on **artificial intelligence, intelligent sensing, and data-driven modeling**.

---

## 📄 Repository Contents

| File | Description |
|------|--------------|
| `EXECUTIVE_SUMMARY.pdf` | One-page English summary of the research proposal |
| `PLAN_DE_12_MESES_DOCTORADO.pdf` | 12-month detailed roadmap for the doctoral development |
| `PREPRINT_DOCTORADO.pdf` | IEEE-style preprint manuscript (simulation + methodology) |
| `demo/demo_preprint_mmi.py` | Python simulation and reproducibility demonstration |
| `figures/` | Contains generated figures used in the preprint |
| `CITATION.cff` | Academic citation metadata |
| `LICENSE` | Project license (MIT) |

---

## 🧪 Figures

![Spectral Shift](figures/figura_espectros_MMI.png)  
*Simulated multimode interference spectral response for different pH values.*

![Calibration](figures/calibracion_pH_lambda.png)  
*Linear correlation between pH and peak wavelength (Δλ/ΔpH ≈ 0.089 nm/pH).*

---

## ⚙️ Demo Simulation

To reproduce the preprint’s spectral calibration experiment:

```bash
cd demo
python demo_preprint_mmi.py
