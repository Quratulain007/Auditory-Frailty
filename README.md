# Frailty Audio Analysis App

This Streamlit app extracts frailty-related voice biomarkers from speech audio.

## Features
- Upload WAV audio
- Plot waveform
- Extract:
  - Zero Crossing Rate
  - Shimmer
  - F1 Formant variability
  - Low-frequency energy ratio
- Download CSV output

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
