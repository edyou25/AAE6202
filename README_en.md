# AAE6202 Homework

| Boeing 747 | Simulation |
| --- | --- |
| ![B747](assets/747.png) | ![Animation](assets/anime.gif) |

Boeing 747 circular-trajectory flight simulation (LQR control + 4th-order RK integration).

## Theory Documents

- [Part 1: LQR (Convex Optimization Background / Quadratic Form)](docs/lqr_en.md)
- [Part 2: RK4 (Continuous Dynamics Discrete Propagation)](docs/rk_en.md)
- [Part 3: Bayes/MAP Estimation (Kalman Filtering)](docs/estimation_en.md)

## Dependencies

```bash
conda env create -f env.yaml
```

## How to Run

```bash
python3 run.py
```

## File Descriptions

- `run.py`: Main entry point — runs the simulation and generates plots.
- `controller.py`: Circular-trajectory tracking controller (outer-loop guidance + inner-loop LQR + speed holding).
- `dynamics.py`: Boeing 747 2D planar dynamics model + RK4 integrator.
- `estimation.py`: Bayes/MAP state estimation module (prior prediction + measurement update).
- `visual.py`: Aircraft point-cloud visualization module (fuselage, wings, tail) and animation export.

## LaTeX Report

- `latex/_main.tex`: Main course-report file.
- `latex/Academic.cls`: Report template class file.
- `latex/abstract.tex`: Abstract content file.
- `latex/references.bib`: Reference database.
- `latex/wordcount.py`: LaTeX body word-count script.
- `latex/template/`: Original template backup.
- `latex/output/_main.pdf`: Generated report PDF.
