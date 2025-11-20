Alloy Property Predictor & Alloy Type Helper

**Live App:**  
 [https://alloypredictormaterialsscience-hs3drbwrc7gtmoltl6v3tx.streamlit.app/](https://alloypredictormaterialsscience-hs3drbwrc7gtmoltl6v3tx.streamlit.app/)

Overview
The **Alloy Property Predictor & Alloy Type Helper** is an interactive **Streamlit-based web application** that predicts key mechanical properties of alloys (such as **yield strength** and **hardness**) using their **elemental composition** and **processing parameters**.  
It also identifies the **alloy family** (e.g., stainless steel, aluminum alloy, brass, bronze) based on composition rules.

This app integrates:
-  Machine Learning (Random Forest) for **regression** and **classification**
- **Automatic model saving/loading** using `joblib`
- **Data visualization** tools for EDA (heatmaps, pairplots, histograms)
   **Alloy Type Helper** for quick material classification
- A built-in **synthetic data generator** for offline testing
Objectives
1. Create a unified interface for alloy property analysis and prediction.  
2. Enable regression (single and multi-output) and classification tasks.  
3. Provide a lightweight, open-source app for materials informatics learning.  
4. Predict multiple mechanical properties simultaneously (e.g., Yield Strength + Hardness).  
5. Identify probable alloy families from chemical compositions.  Features
| Module | Description |
|---------|--------------|
| **Data Upload / Generation** | Upload real alloy data or use a built-in synthetic dataset. |
| **EDA (Exploratory Data Analysis)** | View correlation heatmaps, histograms, and data stats. |
| **Model Training** | Train Random Forest regression or classification models. |
| **Prediction Interface** | Enter alloy features manually and get property predictions. |
| **Model Persistence** | Models saved to disk as `saved_model.pkl` (auto-loads next session). |
| **Alloy Type Helper** | Rule-based alloy type detection from composition. |
 How It Works
1. **Upload CSV** — Provide alloy dataset with columns like `%Cu`, `%Zn`, `%Mg`, `Aging_Temp_C`, etc.  
2. **Or Use Synthetic Data** — Auto-generates 200 random but realistic samples with simulated mechanical properties.  
3. **Explore Data** — Generate correlation heatmaps and statistical summaries.  
4. **Train Models** — Choose Regression or Classification and train a Random Forest model.  
5. **Predict** — Input new feature values and obtain predicted results.  
6. **Alloy Type Helper** — Analyze compositions (e.g., `Fe:70, Cr:12, Ni:8`) and get the alloy family suggestion.

 Synthetic Dataset Description
When no real dataset is uploaded, the app generates a synthetic dataset of 200 entries.

| Feature | Description |
|----------|-------------|
| `Cu`, `Zn`, `Mg`, `Cr`, `Ti` | Elemental composition (%) |
| `Aging_Temp_C` | Artificial aging temperature (°C) |
| `Aging_Time_h` | Aging duration (hours) |
| `Quench_Rate_Cps` | Cooling rate (°C/s) |
| **Targets** | `Yield_Strength_MPa`, `Hardness_HV` (computed via linear model + random noise) |

---

Tech Stack
| Component | Technology |
|------------|-------------|
| Programming Language | Python 3.9+ |
| Framework | Streamlit |
| ML Library | scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Persistence | joblib |
| Environment | VS Code / Anaconda / Streamlit Cloud |

Clone the Repository
```bash
git clone https://github.com/yourusername/alloy-property-predictor.git
cd alloy-property-predictor

