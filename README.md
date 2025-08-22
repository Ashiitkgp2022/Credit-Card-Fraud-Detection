# Credit Card Fraud Detection Streamlit App

This is a web-based application built with Streamlit that predicts whether a credit card transaction is fraudulent or not. The app is based on the credit card fraud detection machine learning project that uses XGBoost as the primary algorithm.

## Features

- **Interactive Interface**: Easy-to-use web interface for entering transaction details
- **Real-time Prediction**: Instant fraud probability calculation
- **Risk Assessment**: Color-coded risk levels (Low, Medium, High)
- **Feature Engineering**: Automatically calculates derived features like distance from home, time-based features, etc.
- **Detailed Analysis**: Shows key risk factors that contribute to the fraud prediction

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open in Browser**:
   The app will automatically open in your default browser at `http://localhost:8501`

## How to Use

1. **Enter Transaction Details**:
   - Transaction amount and category
   - Date and time of transaction
   - Customer information (gender, birth year, job, etc.)
   - Location details (customer and merchant coordinates)

2. **Get Prediction**:
   - Click "Analyze Transaction" button
   - View the fraud probability percentage
   - See the risk level and recommendation
   - Check detailed risk factors analysis

## Input Fields

### Transaction Information
- **Amount**: Transaction amount in USD
- **Category**: Type of merchant (e.g., grocery, entertainment, gas, etc.)
- **Date/Time**: When the transaction occurred

### Customer Information
- **Gender**: M/F
- **Birth Year**: Year of birth (used to calculate age)
- **City Population**: Population of customer's city
- **Job**: Customer's job category
- **State**: Customer's state

### Location Information
- **Customer Latitude/Longitude**: Customer's location coordinates
- **Merchant Latitude/Longitude**: Merchant's location coordinates

## Risk Factors

The model considers several risk factors:

- **High Transaction Amounts**: Larger amounts have higher fraud risk
- **Time-based Patterns**: Night-time and weekend transactions
- **Distance from Home**: Transactions far from customer's location
- **Merchant Category**: Certain categories have higher fraud rates
- **Customer Demographics**: Age and other demographic factors

## Model Information

- **Algorithm**: XGBoost Classifier
- **Training Accuracy**: ~100% (as mentioned in the original notebook)
- **Test Accuracy**: ~99.9%
- **Features**: 14+ engineered features including time-based, location-based, and demographic features

## Risk Levels

- 🟢 **LOW RISK** (< 30%): Transaction appears legitimate - Approve
- 🟡 **MEDIUM RISK** (30-70%): Additional verification recommended - Review
- 🔴 **HIGH RISK** (> 70%): High probability of fraud - Block transaction

## Note

This is a demonstration application based on the credit card fraud detection project. In a production environment, you would:

1. Load an actual pre-trained model instead of the rule-based prediction
2. Implement additional security measures
3. Add more sophisticated feature engineering
4. Include real-time data validation
5. Integrate with actual banking systems

## Files

- `streamlit_app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `Credit Card.ipynb`: Original Jupyter notebook with ML model development
- `README.md`: This documentation file

## Disclaimer

This is a demonstration system for educational purposes. In production, additional security measures, data validation, and compliance with financial regulations would be required.
