# Machine Learning Project: F1 Qualifying

Dieses Machine-Learning-Projekt entstand im Rahmen meines Studiums.

Dieses Projekt untersucht, bereinigt und modelliert Formel-1-Qualifying-Zeiten. Ziel ist es, auf Basis von Trainings-, Qualifying- und Wetterdaten vorherzusagen, wie stark verschiedene Faktoren die Qualifying-Performance beeinflussen.

Das Projekt ist als Notebook-Workflow aufgebaut: von der explorativen Datenanalyse über das Preprocessing bis zum Vergleich mehrerer Machine-Learning-Modelle.

## Inhalt

- Explorative Analyse von Formel-1-Qualifying-Daten
- Zusammenführung von Qualifying-, Trainings- und Wetterdaten
- Bereinigung fehlender Werte und Feature Engineering
- Encoding kategorialer Variablen wie Fahrer, Teams und Grand Prix
- Vergleich verschiedener Modelle:
  - Lineare Regression
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
  - Averaging Ensemble
- Vergleich von chronologischem Split und zufälligem Race-Split

## Projektstruktur

```text
.
├── Daten/
│   ├── f1_qualifying_dataset_2018.csv ... f1_qualifying_dataset_2026.csv
│   ├── f1_weather_dataset_2018.csv ... f1_weather_dataset_2026.csv
│   ├── ML_Data.csv
│   ├── ML_Data_Preprocessed.csv
│   └── ML_Data_Encoded.csv
├── EDA_Preprocessing/
│   ├── EDA.ipynb
│   └── Preprocessing.ipynb
├── ML_Modelle/
│   ├── ML-Projekt_F1_chrono.ipynb
│   ├── ML-Projekt_F1_random.ipynb
│   └── helper.py
├── README.md
└── requirements.txt
```

## Notebooks

| Notebook | Beschreibung |
| --- | --- |
| `EDA_Preprocessing/EDA.ipynb` | Explorative Datenanalyse, Missing Values, Fahrer- und Streckenvergleiche, Korrelationen und Visualisierungen |
| `EDA_Preprocessing/Preprocessing.ipynb` | Datenbereinigung, Mapping von Teams/Fahrern/Grand Prix und Encoding |
| `ML_Modelle/ML-Projekt_F1_chrono.ipynb` | Modelltraining mit chronologischem Race-Split |
| `ML_Modelle/ML-Projekt_F1_random.ipynb` | Modelltraining mit zufälligem Race-Split auf Rennenebene |

## Setup

Voraussetzung: Python 3.10 oder neuer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Unter macOS oder Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Danach können die Notebooks in Jupyter Lab, Jupyter Notebook oder VS Code geöffnet werden.

## Ausführung

Die empfohlene Reihenfolge ist:

1. `EDA_Preprocessing/EDA.ipynb`
2. `EDA_Preprocessing/Preprocessing.ipynb`
3. `ML_Modelle/ML-Projekt_F1_chrono.ipynb`
4. `ML_Modelle/ML-Projekt_F1_random.ipynb`

Die Notebooks suchen den Ordner `Daten` automatisch im aktuellen Verzeichnis oder in einem übergeordneten Projektverzeichnis. Dadurch funktionieren die CSV-Importe sowohl beim Start aus dem Projektroot als auch aus den jeweiligen Unterordnern.

## Daten

Die Daten liegen als CSV-Dateien im Ordner `Daten`. Enthalten sind Qualifying- und Wetterdaten für mehrere Formel-1-Saisons sowie vorbereitete Datensätze für das Modelltraining.

Die wichtigsten Modellierungsdateien sind:

- `Daten/ML_Data.csv`: zusammengeführter Datensatz
- `Daten/ML_Data_Preprocessed.csv`: bereinigter Datensatz
- `Daten/ML_Data_Encoded.csv`: finaler Datensatz für das Modelltraining

## Modelle und Bewertung

In den Modellierungsnotebooks werden mehrere Regressionsmodelle trainiert und miteinander verglichen. Dabei werden zwei Split-Strategien betrachtet:

- Chronologischer Split: Training auf früheren Rennen, Test auf späteren Rennen
- Zufälliger Race-Split: zufällige Aufteilung auf Rennenebene, damit Daten eines Rennens nicht gleichzeitig in Training und Test landen

Die Hilfsfunktionen für diese Splits befinden sich in `ML_Modelle/helper.py`.

## Verwendete Bibliotheken

- `fastf1`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `seaborn`

## Hinweis

Dieses Projekt ist als Machine-Learning-Projekt und Notebook-Dokumentation aufgebaut. Die Ergebnisse hängen von den enthaltenen Datensätzen, der gewählten Split-Strategie und den verwendeten Features ab.
