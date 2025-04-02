## 🔥 Fire Weather Index (FWI) Pipeline

This repository implements a complete pipeline that transforms low-resolution meteorological data into high-resolution data and calculates the Fire Weather Index (FWI) based on the high-resolution outputs.

---

### Project Structure

```
📄 1_Data_preprocessing/
└── calc_mean_std.py         # Compute mean and standard deviation of input data

📄 2_LR_to_HR/
├── configs/                 # Experiment configuration files
├── datasets/                # Dataset loading and preprocessing code
├── logs/                    # Training logs output directory
├── model/                   # Deep learning model definitions
├── utils/                   # Utility functions for training
├── visualization/           # Visualization scripts
├── train.py                 # Training script
├── test.py                  # Evaluation script
├── train.sh / test.sh       # Shell scripts for training/testing

📄 3_calc_FWI/
├── calc_HR_FWI.py           # Calculate FWI from high-resolution data
├── calc_LR_FWI.py           # Calculate FWI from low-resolution data
├── FWIFunctions_v6.py       # Functions for FWI computation
└── visualization/           # FWI visualization tools
```

---

### Execution Flow

1. **Preprocess input data**

   ```bash
   python 1_Data_preprocessing/calc_mean_std.py
   ```

2. **Train resolution enhancement model**

   ```bash
   bash 2_LR_to_HR/train.sh
   ```

3. **Test the model (generate high-resolution predictions)**

   ```bash
   bash 2_LR_to_HR/test.sh
   ```

4. **Compute Fire Weather Index (FWI)**

   - High-resolution FWI:
     ```bash
     python 3_calc_FWI/calc_HR_FWI.py
     ```
   - Low-resolution FWI:
     ```bash
     python 3_calc_FWI/calc_LR_FWI.py
     ```

---

### Required Packages

- Refer to `environment.yaml` for full dependencies

---


