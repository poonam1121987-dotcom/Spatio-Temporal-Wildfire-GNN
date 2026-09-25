# Spatio-Temporal Graph Neural Network for Wildfire Risk Forecasting

A PyTorch Geometric implementation using Graph Convolutional Networks (GCN) to model spatial relationships across Mediterranean climate telemetry (Temperature, Relative Humidity, Wind Speed) for early wildfire detection.

## Model Performance
* **Spatial Accuracy:** 99.6%
* **Precision:** 0.996
* **Recall:** 1.000 (0 missed wildfire risks)
* **F1-Score:** 0.998

## Technical Highlights
* **Feature Normalization:** Applied Min-Max scaling across non-Euclidean node features to ensure stable gradient descent.
* **Reproducibility:** Locked random seeds across PyTorch and NumPy for deterministic graph connectivity and weight initialization.
* **Frameworks:** PyTorch, PyTorch Geometric, Pandas, NumPy, Scikit-Learn
