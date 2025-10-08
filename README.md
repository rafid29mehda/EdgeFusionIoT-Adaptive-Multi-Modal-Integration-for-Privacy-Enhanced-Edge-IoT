# EdgeFusionIoT: Multi-Modal Data Fusion System for IoT and Edge Computing

## Overview

EdgeFusionIoT is a simulated platform for fusing multi-modal data from IoT sensors, performing edge-based processing, and enabling real-time decision-making. It integrates techniques across embedded systems, sensor networks, AI/ML, and communication protocols. 

Key highlights:
- **Sensor Simulation**: Handles time-series (e.g., temperature, GPS) and multimedia (e.g., images).
- **Edge ML**: Uses TensorFlow Lite for on-device inference and federated learning for privacy.
- **Communication**: Simulates protocols with error correction (Reed-Solomon).
- **Data Fusion**: Applies Kalman filters, CNNs, and RNNs for anomaly detection.
- **Extras**: Scalability tests, blockchain logging, and visualizations.

This is a Python-based simulation (no real hardware required), making it easy to run in environments. Extendable to real devices, such as Raspberry Pi.

## Features and Techniques

- **Sensor Integration**: Virtual sensors generating diverse data types.
- **Edge Processing**: ML models (RNN for sequences, CNN for images) converted to TFLite.
- **Communication Layer**: Socket-based simulation with error correction.
- **Analytics and Fusion**: Multimodal algorithms for event prediction (e.g., environmental anomalies).
- **Federated Learning**: Privacy-preserving aggregation using Flower.
- **Blockchain**: Secure logging simulation via Web3.
- **Visualizations**: Matplotlib plots of data streams and fusion results.
- **Scalability**: Timed tests for larger datasets.

