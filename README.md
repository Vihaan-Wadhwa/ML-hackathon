# 🗺️ GPS Route Reconstruction Using Machine Learning and Road Network Analysis

> An intelligent GPS map-matching and route reconstruction system that combines machine learning, road network topology, probabilistic transition modelling and Viterbi decoding to reconstruct the most likely route from GPS observations.

## 📌 Project Overview

GPS devices frequently produce noisy, sparse or inaccurate location observations. Simply connecting consecutive GPS coordinates or selecting the nearest road segment can result in incorrect routes, especially around intersections, parallel roads and complex road networks.

This project addresses the problem through a multi-stage GPS route reconstruction pipeline.

The system processes raw GPS observations, predicts candidate road segments using a trained machine learning model, evaluates transitions between road segments using road network connectivity, and reconstructs the most probable continuous route.

The architecture is divided into four devices that work together to produce the final reconstructed route.

## 🎯 Objectives

- Process and prepare raw GPS trajectory data.
- Extract relevant features from GPS observations.
- Identify possible road-segment candidates.
- Predict candidate relevance using machine learning.
- Construct a directed road network graph.
- Calculate network transition distances and probabilities.
- Apply Hidden Markov Model-based sequence decoding.
- Reconstruct continuous and geographically meaningful routes.
- Evaluate reconstructed routes against ground-truth data.

## 🏗️ System Architecture

The project consists of four major devices.

### Device 1 — GPS Data Processing and Feature Extraction

Responsible for preparing raw GPS observations and generating structured information for subsequent processing.

Key responsibilities:

- GPS coordinate and timestamp processing.
- Data validation and preprocessing.
- Feature extraction.
- Candidate road-segment preparation.
- Structured output generation.

**Output:** Processed GPS observations and candidate information.

### Device 2 — Machine Learning Candidate Prediction

Uses a trained machine learning model to estimate the likelihood that a candidate road segment corresponds to a GPS observation.

**Model:** HistGradientBoostingClassifier

Key responsibilities:

- Load the trained model artifact.
- Process the candidate feature dataset.
- Generate candidate predictions.
- Calculate prediction probabilities.
- Export candidate prediction results for route reconstruction.

Current observed configuration:
- 12 input features.
- Model artifact: `models/final_model.joblib`.
- Evaluator prediction dataset: 116,985 candidate rows.

**Output:** Candidate predictions and probability scores.

### Device 3 — Road Network and Route Reconstruction

This device performs the core route reconstruction process.

It combines candidate probabilities with road network topology to identify the most likely sequence of road segments.

Key components:

**1. Road Graph Construction**
- Load road nodes and directed arcs.
- Construct the road network graph.
- Preserve original arc identifiers.
- Calculate road-edge distances.

**2. GPS Displacement**
- Calculate geographical displacement between consecutive GPS observations.
- Handle invalid coordinates and timestamp conditions.

**3. Network Transition Distance**
- Calculate shortest-path distances between candidate road segments.
- Account for directed road connectivity.
- Handle disconnected roads and parallel arcs.

**4. Transition Probability Model**
- Calculate transition probabilities.
- Apply effective beta and GPS displacement scaling.
- Normalize transition matrices.

**5. Viterbi Decoding**
- Evaluate candidate sequences.
- Use emission and transition probabilities.
- Select the most likely continuous road sequence.
- Handle missing observations and disconnected transitions.

**6. Route Projection**
- Convert reconstructed road sequences into GPS coordinates.
- Generate the final reconstructed route output.

**Testing status:** 130 Device 3 tests passed in the latest reported test run.

### Device 4 — Backend Integration and Visualization

Responsible for connecting the individual components and presenting the reconstructed results.

Key responsibilities:

- Backend integration.
- Model and prediction loading.
- Route reconstruction integration.
- API-based access to processing functionality.
- Dashboard and result visualization.
- Displaying reconstructed GPS trajectories.

The final integration should expose both the predicted route and its evaluation against reference routes.

## 🔄 End-to-End Workflow

```text
              RAW GPS OBSERVATIONS
                       |
                       v
              +------------------+
              |     DEVICE 1     |
              | Data Processing  |
              | Feature Extraction|
              +------------------+
                       |
                       v
              +------------------+
              |     DEVICE 2     |
              | Machine Learning |
              | Candidate Scores |
              +------------------+
                       |
                       v
              +------------------+
              |     DEVICE 3     |
              | Road Graph       |
              | Transition Model |
              | Viterbi Decoding |
              +------------------+
                       |
                       v
              +------------------+
              |     DEVICE 4     |
              | Backend & UI     |
              | Visualization    |
              +------------------+
                       |
                       v
               RECONSTRUCTED ROUTE
                       |
                       v
               GROUND-TRUTH EVALUATION
```

## 🧰 Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| ML Model | HistGradientBoostingClassifier |
| Data Processing | Pandas, NumPy |
| Road Network | NetworkX |
| Route Reconstruction | Hidden Markov Model, Viterbi Algorithm |
| Data Storage | Parquet, Joblib |
| Testing | Pytest |
| Backend | Python backend / API integration |
| Visualization | Dashboard and geographic route visualization |
| Version Control | Git and GitHub |

## 📁 Repository Structure

The following represents the major project components.

```text
GPS-ROUTE-RECONSTRUCTION/
│
├── backend/
│   └── predictor.py
│
├── modules/
│   └── device3/
│       ├── graph/
│       │   └── graph_builder.py
│       │
│       ├── decoder/
│       │   ├── candidate_io.py
│       │   ├── reconstruction.py
│       │   ├── route_projection.py
│       │   ├── transition_model.py
│       │   └── viterbi.py
│       │
│       ├── visualization/
│       │   └── road_network_plot.py
│       │
│       └── tests/
│
├── models/
│   └── final_model.joblib
│
├── data/
│   └── evaluator_test/
│
├── results/
│   ├── candidate_predictions_evaluator.parquet
│   └──
│
├── requirements.txt
└── README.md
```

*Note: This is a representative structure of the observed project components, not an exhaustive file listing.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vihaan-Wadhwa/ML-hackathon.git
cd ML-hackathon
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Verify the Device 3 package

```powershell
python -m compileall -q modules/device3
```

```powershell
python -c "import modules.device3; print('DEVICE 3 IMPORT SUCCESSFUL')"
```

## 🚀 Running the Prediction Pipeline

The candidate prediction function can be executed using:

```powershell
python -c "from backend.predictor import predict_candidates; print(predict_candidates('results/candidate_predictions_evaluator.parquet', 'results/integration_smoke_test.parquet'))"
```

This processes the evaluator candidate dataset and writes the resulting predictions to the specified output Parquet file.

Ensure that the trained model and input dataset are available at the expected paths.

## 🧪 Testing

The project uses Pytest for automated testing.

Run the Device 3 test suite:

```powershell
python -m pytest modules/device3/tests -v
```

Run the complete test suite:

```powershell
python -m pytest -v
```

Latest reported Device 3 test result:

| Test Parameter | Result |
|---|---:|
| Tests executed | 130 |
| Tests passed | 130 |
| Tests failed | 0 |
| Execution time | 8.05 seconds |
| Test pass rate | 100% |

**Important:** The test pass rate measures software test success. It does not represent the route reconstruction model's prediction accuracy.

## 📊 Model Evaluation

A complete evaluation must compare predicted results with labelled ground-truth road segments and reference routes.

The following metrics are planned for final evaluation.

### Candidate Classification Metrics

| Metric |
