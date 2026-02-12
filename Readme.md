# 📞 Customer Churn Predictor

A machine learning-powered web application to predict customer churn for telecom companies. Built with Streamlit and trained on the Telco Customer Churn dataset, this tool helps identify customers at risk of leaving and provides actionable insights for retention strategies.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **Interactive Dashboard**: User-friendly Streamlit interface for real-time churn predictions
- **Multiple ML Models**: Trained and compared 4 different models (Logistic Regression, Random Forest, AdaBoost, XGBoost)
- **Feature Engineering**: Enhanced prediction accuracy with custom-engineered features
- **Risk Assessment**: Clear visualization of churn probability with actionable recommendations
- **Comprehensive Analysis**: Exploratory Data Analysis (EDA) and model evaluation included

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dontknow492/Customer-Churn-Predictor.git
   cd "Customer Churn Predictor"
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib xgboost
   ```

3. **Run the application**
   ```bash
   streamlit run main.py
   ```

4. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset which contains:
- **7,043 customers** with 21 features
- **Customer demographics**: Gender, age, partner status, dependents
- **Service information**: Phone, internet, streaming, tech support
- **Account details**: Contract type, payment method, tenure, charges
- **Target variable**: Churn (Yes/No)

## 🔬 Models

Four machine learning models were trained and evaluated:

| Model | Accuracy | Best For |
|-------|----------|----------|
| **Logistic Regression** | ~75% | Production (deployed) |
| **Random Forest** | ~75% | Feature importance analysis |
| **AdaBoost** | ~73% | Ensemble comparison |
| **XGBoost** | ~74% | Gradient boosting benchmark |

The **Logistic Regression** model was selected for deployment due to its:
- Fast inference time
- Interpretability
- Comparable accuracy to more complex models
- Lower resource requirements

## 🏗️ Project Structure

```
Customer Churn Predictor/
├── main.py                          # Streamlit web application
├── Readme.md                        # Project documentation
├── dataset/
│   └── Telco-Customer-Churn.csv    # Training dataset
├── models/
│   ├── logistic/
│   │   └── model.joblib            # Deployed model
│   ├── random_forest/
│   │   ├── model.joblib
│   │   └── classification_report.txt
│   ├── ada/
│   │   └── model.joblib
│   └── xgboost/
│       └── model.joblib
└── src/
    ├── eda.ipynb                   # Exploratory Data Analysis
    └── train.ipynb                 # Model training & evaluation
```

## 💡 Usage

### Web Interface

1. **Enter Customer Details**: Fill in demographic information (gender, senior citizen status, partner, dependents, tenure)

2. **Add Financial Data**: Input monthly charges and total charges

3. **Select Services**: Choose the services the customer subscribes to (phone, internet, streaming, tech support, etc.)

4. **Choose Contract Type**: Select contract duration and payment method

5. **Predict**: Click the "Predict Churn Risk" button to get results

### Sample Prediction

The model outputs:
- **Churn Probability**: Percentage likelihood of customer leaving
- **Risk Level**: 
  - ✅ **Low Risk** (<50%): Customer likely to stay
  - ⚠️ **High Risk** (>50%): Customer likely to churn
- **Recommendation**: Suggested action for high-risk customers

## 🔧 Development

### Running Notebooks

To explore the data or retrain models:

```bash
jupyter notebook src/eda.ipynb
jupyter notebook src/train.ipynb
```

### Training New Models

1. Open `src/train.ipynb`
2. Run all cells to train models on the dataset
3. Models will be saved in the `models/` directory
4. Update `main.py` to load your preferred model

### Feature Engineering

The model includes a custom feature:
- **Total_Services**: Count of services subscribed by the customer
  - Improves prediction accuracy by capturing service engagement

## 📈 Model Performance

### Random Forest Classification Report
```
              precision    recall  f1-score   support
           0       0.90      0.75      0.82      1053
           1       0.50      0.76      0.60       352
    accuracy                           0.75      1405
```

**Key Insights**:
- High precision (90%) for predicting non-churners
- Good recall (76%) for identifying potential churners
- 75% overall accuracy on test set

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib
- **Visualization**: Matplotlib, Seaborn (in notebooks)

## 📝 Future Enhancements

- [ ] Add model retraining interface
- [ ] Implement batch prediction from CSV upload
- [ ] Add more advanced feature engineering
- [ ] Include SHAP values for model explainability
- [ ] Deploy to cloud (Streamlit Cloud, AWS, or Azure)
- [ ] Add customer lifetime value (CLV) prediction
- [ ] Implement A/B testing for retention strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Telco Customer Churn dataset from [IBM Sample Data Sets](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Streamlit for the amazing web framework
- Scikit-learn community for excellent documentation

## 📧 Contact

For questions or feedback, please open an issue or contact [your-email@example.com]

---

**Made with ❤️ for better customer retention**

