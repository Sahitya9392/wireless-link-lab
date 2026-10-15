# Wireless Link Lab 📡

A Monte Carlo simulation tool built with Python and Streamlit to analyse the bit error rate (BER) of BPSK modulation over various wireless fading channels and transmission techniques.

**[Live Demo: Try the app here!](https://wireless-link-lab-umrwa2ruqjjm9nfutrqvdd.streamlit.app)**

## Features
* **Fading Channel Models:** Compare Gamma, Rayleigh, Weibull, and Gaussian (line-of-sight) fading channels under unit average power.
* **Receive Diversity:** Simulate a 1×2 SIMO system using Maximal Ratio Combining (MRC) to observe diversity gain.
* **OFDM Transmission:** Evaluate multi-carrier transmission over a multipath channel with a cyclic prefix and one-tap equalisation.
* **Spatial Multiplexing (MIMO):** Analyse a 2×2 MIMO system using a zero-forcing receiver to double the data rate.
* **Forward Error Correction:** Observe coding gains using a Hamming (7,4) block code with hard-decision decoding.
* **Interactive Visualisations:** Plot BER curves against theoretical AWGN/Rayleigh limits, view fading envelopes, probability density functions, and explore 3D Weibull BER surfaces using Plotly.

## Running Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Sahitya9392/wireless-link-lab.git](https://github.com/Sahitya9392/wireless-link-lab.git)
   cd wireless-link-lab
