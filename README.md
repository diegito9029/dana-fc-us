# DANA Automation Data Scientist Take Home Test: Feature Creation + User Segmentation

This project contains the full workflow and documentation for engineering user-level behavioural features from transaction data and performing segmentation using K-Means clustering. It transforms raw transaction logs into interpretable behavioural, psychographic, and lifestyle features, and then identifies meaningful user segments through unsupervised learning.

Feel free to inspect each part, run it locally, and modify as needed. [Click here to learn more about the project](https://github.com/verneylmavt/dana-fc-us/blob/1c58b32aec71444be3f083d61a7f331d5cb3385a/data/test.txt).

## 📁 Project Structure

```
dana-fc-us
│
├─ data/
│  ├─ test.txt
│  └─ transactions.csv
│
├─ user_features_segmentation_code.ipynb
├─ user_features_segmentation_report.docx
├─ README.md
└─ requirements.txt
```

- `user_features_segmentation_code.ipynb`: Contains the complete Python code for data cleaning, feature engineering, derived behavioural scoring, and user segmentation (K-Means + PCA visualization).
- `user_features_segmentation_report.docx`: A detailed analytical report explaining the reasoning, methodology, and findings for all engineered features and the segmentation results.

## ⚙️ Local Setup

0. Make sure to have the prerequisites:

   - Python
   - Conda or venv

1. Clone the repository:

   ```bash
    git clone https://github.com/verneylmavt/dana-fc-us.git
    cd dana-fc-us
   ```

2. Create environment and install dependencies:

   ```bash
   conda create --name dana python=3.10
   conda activate dana

   pip install -r requirements.txt
   ```

3. Open the `user_features_segmentation_code.ipynb` in Jupyter Notebook:
   ```bash
   jupyter notebook user_features_segmentation_code.ipynb
   ```
4. Run all cells sequentially from top to bottom  
   The notebook is designed so that each section depends on the previous one.
5. Review outputs
6. Refer to the report  
   For interpretation, assumptions, and insights, read `user_features_segmentation_report.docx`.
