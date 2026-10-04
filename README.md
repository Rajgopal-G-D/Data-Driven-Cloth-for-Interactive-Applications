# Data-Driven Clothing for Interactive Applications

UE24CS352A Machine Learning - Mini Project
Team: <Rajgopal Deshpande PES2UG24CS395>, <Rohan Rajeev Chirbi PES2UG24CS408>
Problem statement no.: <26>

Based on: Xue & Wu, "Data-Driven Clothing for Interactive Applications".

## Overview
Predict loose-clothing (coat) vertex offsets from 14 joint-rotation quaternions
(56-d input) using PCA + a fully-connected neural network, compared against a
linear-regression baseline.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python cloth_pipeline.py --epochs 200
```
Outputs a results table in the terminal and `results.json`.

## Dataset
(To be filled: synthetic stand-in generated in `make_dataset()`; shapes match the paper.)

## Results
(To be filled after Step 5.)
