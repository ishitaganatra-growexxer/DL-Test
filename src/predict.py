import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def extract_date_parts(date):
    """Handles inconsistent date formats found in the dataset."""
    try:
        if '-' in str(date):
            parts = str(date).split('-')
            return int(parts[0]), int(parts[1]), int(parts[2])
        elif '/' in str(date):
            parts = str(date).split('/')
            return int(parts[2]), int(parts[1]), int(parts[0])
    except:
        return 2020, 1, 1
    return 2020, 1, 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to test.csv')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to save results')
    args = parser.parse_args()

    # 1. Load Data
    # We keep df_original to preserve all original columns and values for the final output
    df_original = pd.read_csv(args.input)
    df = df_original.copy()

    # 2. Preprocessing (Strictly following solution.ipynb logic)
    df.drop('patient_id', axis=1, inplace=True)
    
    # Date extraction
    df[['year','month','day']] = df['admission_date'].apply(lambda x: pd.Series(extract_date_parts(x)))
    df.drop('admission_date', axis=1, inplace=True)

    # Outlier capping and median imputation (using values from training set)
    df.loc[df['age'] > 100, 'age'] = 53.0 
    df['age'] = df['age'].fillna(53.0)
    df['glucose_level_mgdl'] = df['glucose_level_mgdl'].fillna(103.6)

    # Skewed feature transformation
    for col in ['length_of_stay_days', 'creatinine_mgdl', 'prior_admissions_1yr']:
        df[col] = np.log1p(df[col])

    # Categorical Encoding
    cat_cols = ['gender', 'discharge_day_of_week', 'insurance_type']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Alignment: Fixed to exactly 25 features to match model expectations
    expected_cols = [
        'age', 'admission_type', 'discharge_destination', 'length_of_stay_days',
        'charlson_comorbidity_index', 'prior_admissions_1yr', 'n_medications_discharge',
        'glucose_level_mgdl', 'blood_pressure_systolic', 'sodium_meql',
        'creatinine_mgdl', 'haemoglobin_gdl', 'year', 'month', 'day',
        'gender_M', 'discharge_day_of_week_Mon', 'discharge_day_of_week_Sat',
        'discharge_day_of_week_Sun', 'discharge_day_of_week_Thu',
        'discharge_day_of_week_Tue', 'discharge_day_of_week_Wed',
        'insurance_type_Medicaid', 'insurance_type_Medicare', 'insurance_type_Private'
    ]
    
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    
    df = df[expected_cols] 

    # 3. Scaling
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 4. Inference
    try:
        model = tf.keras.models.load_model('model.h5')
        probs = model.predict(df).flatten()
        # Using the best threshold (0.57) found in the notebook
        preds = (probs > 0.57).astype(int) 
    except Exception as e:
        print(f"Error: {e}. Ensure 'model.h5' exists in the directory.")
        return

    # 5. Save Results (Merging predictions back to the original test data)
    df_original['readmitted_30d'] = preds
    
    df_original.to_csv(args.output, index=False)
    print(f"Successfully saved full data with predictions to {args.output}")

if __name__ == "__main__":
    main()