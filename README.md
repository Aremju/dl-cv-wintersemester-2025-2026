# Deep Learning for Computer Vision — Wintersemester 2025/2026

## Überblick

Dieses Repository dokumentiert die Projektarbeit im M.Sc. Informatik-Modul Deep Learning for Computer Vision 
im Wintersemester 2025/2026 von Leon Unruh und Julius Emil Arendt an der Hochschule Bielefeld.
Das Projekt gliedert sich in drei aufeinander aufbauende Meilensteine.

## Projektstruktur

```txt
.
├── data/ (nicht im Repo enthalten, siehe .gitignore)
├── models/ (nicht im Repo enthalten, siehe .gitignore)
├── util/
│ └── ** (Wiederverwendbare Hilfsfunktionen für alle Meilensteine)
├── task/
│ ├── milestone_1/
│ │ └── ** (Meilenstein 1 Quellcode/Notebooks)
│ ├── milestone_2/
│ │ └── ** (Meilenstein 2 Quellcode/Notebooks)
│ └── milestone_3/
│   └── ** (Meilenstein 3 Quellcode/Notebooks)
├── requirements.txt (Python-Abhängigkeiten für Conda/Jupyter)
├── .gitignore
└── README.md
```

---
## Einrichtung

Das Projekt basiert auf Conda und Jupyter Notebooks.

### Conda-Umgebung erstellen

```bash
conda create -n dlcv_project python=3.12
conda activate dlcv_project
```

### Abhängigkeiten installieren

Füge deine benötigten Pakete in `requirements.txt` hinzu und installiere sie mit:

```bash
pip install -r requirements.txt
```

### Jupyter Notebook starten

```bash
jupyter notebook
```

Danach können die Notebooks unter `task/` direkt ausgeführt werden.

## Informationen zu den Autoren des Repositories

### Leon Unruh

- Matrikelnummer: 1286267
- GitHub: [refiver](https://github.com/refiver)
- Studiengang: M.Sc. Informatik

### Julius Emil Arendt

- Matrikelnummer: 1284263
- GitHub: [Aremju](https://github.com/Aremju)
- Studiengang: M.Sc. Informatik