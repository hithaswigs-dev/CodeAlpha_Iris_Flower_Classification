# Iris Flower Classification

## Project Overview

This project is part of the **CodeAlpha Data Science Internship**.

The objective of this project is to classify Iris flowers into three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The classification is performed using a **Random Forest Classifier** based on flower measurements.

---

## Dataset

The dataset contains **150 flower samples** with the following features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The target variable is:

- Species

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Machine Learning Model

The project uses the **Random Forest Classifier** to classify Iris flowers into three species.

---

## Steps Performed

1. Loaded the dataset.
2. Explored the dataset using Pandas.
3. Checked for missing values.
4. Visualized the relationships between features using Seaborn.
5. Encoded the target labels.
6. Split the dataset into training and testing sets.
7. Trained the Random Forest Classifier.
8. Predicted the flower species.
9. Evaluated the model using:
   - Accuracy Score
   - Classification Report
   - Confusion Matrix

---

## Results

- Dataset Size: **150 samples**
- Training Data: **120 samples**
- Testing Data: **30 samples**
- Model Accuracy: **90%**

The model successfully classified Iris flower species with **90% accuracy**.

---

## How to Run

### 1. Clone the repository (if downloaded from GitHub)

```bash
git clone <repository-url>
cd CodeAlpha_Iris_Flower_Classification
```

### 2. Create a virtual environment (Skip this if `venv` is already included)

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows (PowerShell)**

```bash
venv\Scripts\activate
```

**Windows (Command Prompt)**

```bash
venv\Scripts\activate.bat
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 5. Run the project

```bash
python iris_classification.py
```