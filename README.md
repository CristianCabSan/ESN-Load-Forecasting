# 🏠 Energy Consumption Prediction Using Echo State Networks (ESN)

This project is part of my Bachelor's Thesis (TFG) and focuses on **predicting household energy consumption** using **Echo State Networks (ESNs)** — a simple and cost-effective type of recurrent neural network.

## 💡 Why ESNs?

- ✅ **Low computational cost** — ideal for low-resource devices.
- 🧠 **Simplified training** — only the output layer is trained.
- 🔁 **Efficient for time series** — perfect for modeling energy patterns.

## 🔍 Optimization with PSO

To improve the ESN's performance, we used **Particle Swarm Optimization (PSO)** to automatically tune its hyperparameters, ensuring:

- ⚙️ Better accuracy  
- 📉 Reduced error  
- ⏱️ Faster convergence  

## 🏡 Domestic Context

The model is designed to work in the **household context**, where energy consumption is often non-stationary, irregular, and device-dependent. This makes **ESNs a strong candidate** for practical smart home applications.

```## 📁 Structure
📦 ESN-Load-Forecasting
├── 📂 src/                        # Project source code
│   ├── 📂 main/                   # Main training and evaluation scripts
│   ├── 📂 optimization/           # Hyperparameter optimization algorithms
│   ├── 📂 utils/                  # Dataset preprocessing and format handling utilities
└── 📂 resources/                  # Datasets used (not included due to large file size)
