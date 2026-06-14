# 🛒 Shopper Spectrum

## Project Structure
```
shopper_spectrum/
├── shopper_spectrum.ipynb   ← Full analysis notebook (executed)
├── app.py                   ← Streamlit web application
├── ecommerce_data.xlsx      ← Dataset
└── models/
    ├── kmeans_model.pkl     ← Trained KMeans (k=4)
    ├── scaler.pkl           ← StandardScaler for RFM
    ├── label_map.pkl        ← Cluster → Segment label mapping
    ├── similarity_df.pkl    ← Product cosine-similarity matrix
    └── rfm_segments.csv     ← RFM table with segment labels
```

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit openpyxl
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

### 3. Open the Notebook
Open `shopper_spectrum.ipynb` in Jupyter / VS Code.

## Segments
| Segment | Characteristics |
|---------|----------------|
| 💎 High-Value | Recent, frequent, high spend |
| ⭐ Regular | Moderate frequency & spend |
| 🛍️ Occasional | Infrequent, low spend |
| ⚠️ At-Risk | Long inactive, low engagement |
