# Spatio-Temporal Tumour Detection

### Learning Tumour Evolution Through Space, Time, and Deep Visual Representation

Spatio-Temporal Tumour Detection is a deep learning framework for analyzing tumour progression across sequential medical imaging data. Rather than treating each scan as an isolated classification problem, the system approaches tumour analysis as a **spatio-temporal inference problem**, where spatial morphology and temporal evolution jointly describe the underlying progression of a lesion.

The core objective is to extract meaningful representations from imaging sequences and use them to identify, analyze, and characterize changes in tumour structure over time.

---

##  Core Idea

A single medical scan provides a snapshot.

A sequence of scans provides a **trajectory**.

Tumour progression is inherently dynamic: morphology can change, lesions can expand or regress, boundaries can shift, and local structural changes can emerge across successive observations.

This project therefore frames tumour analysis as:

```text
                  Longitudinal Imaging
                         │
                         ▼
              ┌─────────────────────┐
              │ Spatial Representation│
              │                     │
              │ Morphology           │
              │ Texture              │
              │ Structural Features  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Temporal Modeling    │
              │                     │
              │ Progression          │
              │ Growth Dynamics      │
              │ Inter-scan Change    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Spatio-Temporal      │
              │ Representation       │
              │                     │
              │ Spatial + Temporal   │
              │ Feature Interaction  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Tumour Inference    │
              │                     │
              │ Detection            │
              │ Growth Analysis      │
              │ Progression Signals  │
              └──────────┬──────────┘
                         │
                         ▼
                  Clinical Insight
```

The central premise is simple:

> **Tumour progression should be modeled as a dynamic process, not a collection of independent images.**

---

##  Problem Formulation

Conventional image-based tumour detection primarily operates on individual observations:

```text
Image₁ → Model → Prediction₁
Image₂ → Model → Prediction₂
Image₃ → Model → Prediction₃
```

This discards potentially important temporal structure.

Spatio-Temporal Tumour Detection instead considers a sequence:

```text
Image₁ ──→ Image₂ ──→ Image₃ ──→ ... ──→ Imageₙ
   │          │          │                  │
   ▼          ▼          ▼                  ▼
Spatial     Spatial    Spatial            Spatial
Features    Features   Features           Features
   │          │          │                  │
   └──────────┴──────────┴──────────────────┘
                       │
                       ▼
              Temporal Evolution
                       │
                       ▼
            Progression Representation
                       │
                       ▼
              Tumour Analysis
```

This enables the model to reason not only about **what a tumour looks like**, but about **how its observed characteristics change over time**.

---

##  System Architecture

The pipeline is organized around four primary stages:

### 1. Imaging Input

Sequential medical imaging studies are ingested as the temporal observation space.

The input pipeline is responsible for preparing imaging data for downstream representation learning and inference.

```text
Raw Imaging
     │
     ▼
Data Loading
     │
     ▼
Preprocessing
     │
     ▼
Temporal Organization
     │
     ▼
Model Input
```

---

### 2. Spatial Feature Learning

Each observation is transformed into a learned representation of its visual structure.

The spatial representation can capture characteristics such as:

* tumour morphology
* local visual structure
* spatial patterns
* appearance variation
* structural changes
* lesion-level visual signatures

Conceptually:

```text
Iₜ → fθ(Iₜ) → Zₜ
```

where:

* `Iₜ` represents the imaging observation at time `t`
* `fθ` represents the learned spatial feature extractor
* `Zₜ` represents the resulting latent representation

---

### 3. Temporal Representation

The sequence of spatial representations is treated as a temporal signal:

```text
Z₁, Z₂, Z₃, ..., Zₜ
```

Rather than evaluating each representation independently, temporal dependencies can be analyzed to identify patterns associated with:

* progressive growth
* morphological change
* persistence
* emergence
* regression
* temporal irregularities

The resulting representation captures the **evolutionary trajectory** of the observed tumour.

---

### 4. Tumour Analysis & Inference

The learned representation is used to perform downstream analysis and prediction.

The inference pipeline is designed to support:

```text
Detection
   │
   ├── Tumour presence
   │
   ├── Growth characteristics
   │
   ├── Temporal changes
   │
   └── Progression signals
```

The application layer exposes the trained system through a dedicated inference interface for interactive analysis.

---

#  Mathematical View

The problem can be represented as learning a function over a sequence of observations:

```text
X = {x₁, x₂, ..., xₜ}
```

where each `xₜ` represents an imaging observation at time `t`.

A spatial encoder produces:

```text
zₜ = fθ(xₜ)
```

The temporal component then models the evolution:

```text
hₜ = gφ(z₁, z₂, ..., zₜ)
```

The final representation:

```text
hₜ
```

contains information derived from both **spatial morphology** and **temporal context**.

The downstream prediction function can then be represented as:

```text
ŷ = ψω(hₜ)
```

This decomposition separates:

**visual representation → temporal reasoning → tumour inference**

and allows the system to be extended with different architectures without changing the overall problem formulation.

---

#  Why Spatio-Temporal Modeling?

Tumour progression contains information that cannot necessarily be recovered from a single frame.

Consider two observations:

```text
Patient A

Scan 1        Scan 2
  ○             ○
 Stable       Stable
```

versus:

```text
Patient B

Scan 1        Scan 2
  ○             ◯
 Stable       Enlarging
```

The individual images may appear similar in isolation.

The **temporal relationship** between them provides additional information.

This makes longitudinal modeling particularly valuable for identifying changes that emerge only when observations are considered collectively.

---

#  Pipeline

```text
┌──────────────────────────────────────────────┐
│              Medical Imaging Data            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│            Data Preprocessing                 │
│                                              │
│  normalization • formatting • organization   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│         Spatial Representation Learning      │
│                                              │
│     image → latent morphological features    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          Temporal Representation             │
│                                              │
│   sequential features → progression signal   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          Spatio-Temporal Fusion              │
│                                              │
│     spatial morphology + temporal dynamics   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│            Tumour Inference                  │
│                                              │
│ detection • growth • progression • analysis │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              Visualization / UI              │
└──────────────────────────────────────────────┘
```

---

#  Repository Structure

```text
Spatio-Temporal-Tumour-Detection/
│
├── src/
│   ├── main.py
│   └── app.py
│
├── notebooks/
│   └── experimentation/
│
├── data/
│   └── dataset/
│
├── outputs/
│   └── experiments/
│
├── requirements.txt
└── README.md
```

### `src/`

Core implementation containing the training, prediction, analysis, and application pipeline.

### `notebooks/`

Experimental environment for exploratory analysis, visualization, model development, and evaluation.

### `data/`

Expected location for the imaging dataset.

The dataset is intentionally not included in the repository.

### `outputs/`

Reserved for generated predictions, experiment artifacts, logs, and model outputs.

---

#  Technology Stack

```text
Python
│
├── Deep Learning
├── Computer Vision
├── Medical Image Analysis
├── Sequential Modeling
└── Scientific Computing
```

The architecture is intentionally modular so that the underlying representation-learning and temporal components can be evolved independently.

---

#  Installation

Clone the repository:

```bash
git clone https://github.com/harshg1103/Spatio-Temporal-Tumour-Detection.git

cd Spatio-Temporal-Tumour-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the required dataset inside:

```text
data/
```

---

#  Training

Run the training pipeline:

```bash
python src/main.py
```

The training pipeline is responsible for loading the prepared dataset, constructing the learning pipeline, training the model, and generating experiment outputs.

---

#  Inference & Application

Launch the application interface:

```bash
python src/app.py
```

The application provides an interface for interacting with the trained model and performing tumour analysis.

---

#  Experimental Workflow

The project is structured around an iterative research workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
Representation Learning
   ↓
Temporal Modeling
   ↓
Training
   ↓
Inference
   ↓
Evaluation
   ↓
Visualization
   ↓
Iteration
```

This allows experimental architectures and temporal modeling strategies to be evaluated without restructuring the entire pipeline.

---

#  Future Research Directions

The architecture can be extended toward more sophisticated longitudinal medical-imaging systems, including:

* 3D volumetric representation learning
* transformer-based temporal modeling
* cross-scan registration and alignment
* lesion-level segmentation
* multimodal imaging fusion
* uncertainty-aware prediction
* attention-based temporal attribution
* patient-specific progression trajectories
* longitudinal representation learning
* self-supervised pretraining
* weakly supervised tumour localization
* explainable model predictions
* probabilistic progression forecasting

A particularly interesting direction is combining learned representations with explicit biological or geometric constraints to create models that are not only predictive, but capable of representing **plausible tumour evolution trajectories**.

---

#  Disclaimer

This repository is intended for research and educational purposes.

Predictions generated by the system should **not** be interpreted as medical diagnoses or used as a substitute for evaluation by qualified medical professionals.

---

#  Project Status

This project is an active research/engineering prototype focused on exploring deep learning approaches to spatio-temporal tumour analysis.

The dataset and generated experiment artifacts are not included in the repository.

---

## Core Research Question

> **Can deep learning move beyond detecting what a tumour looks like and learn how that tumour evolves?**

Spatio-Temporal Tumour Detection explores that question by treating longitudinal medical imaging as a **dynamic sequence of biological observations**, combining spatial representation learning with temporal reasoning to transform static scans into structured models of tumour progression.
