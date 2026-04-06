import pandas as pd
import os

def process_bureau(bureau_path):
    print("Processing Bureau data...")
    bureau = pd.read_csv(bureau_path)
    
    # Aggregate past credit behavior per user
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count', # Total number of previous loans
        'DAYS_CREDIT': ['min', 'max', 'mean'], # Recency of credit
        'AMT_CREDIT_SUM': ['sum', 'mean'] # Total credit amount
    })
    
    # Flatten the multi-level index columns created by pandas agg()
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).strip().upper() for col in bureau_agg.columns.values]
    return bureau_agg.reset_index()

def process_installments(installments_path):
    print("Processing Installments data...")
    installments = pd.read_csv(installments_path)
    
    # Calculate a late payment flag (1 if late, 0 if on time)
    installments['LATE_PAYMENT'] = (installments['DAYS_ENTRY_PAYMENT'] > installments['DAYS_INSTALMENT']).astype(int)
    
    # Aggregate installment behavior
    inst_agg = installments.groupby('SK_ID_CURR').agg({
        'LATE_PAYMENT': ['sum', 'mean'], # Total late payments and percentage of late payments
        'AMT_PAYMENT': ['sum', 'mean']
    })
    
    inst_agg.columns = ['INST_' + '_'.join(col).strip().upper() for col in inst_agg.columns.values]
    return inst_agg.reset_index()

def main():
    # Define file paths based on our DVC hierarchy
    app_path = "data/interim/application_masked.csv"
    bureau_path = "data/raw/bureau.csv"
    installments_path = "data/raw/installments_payments.csv"
    output_path = "data/processed/train_final.csv"

    print("Loading masked application data...")
    app_df = pd.read_csv(app_path)

    # 1. Process and Merge Bureau Data
    if os.path.exists(bureau_path):
        bureau_agg = process_bureau(bureau_path)
        app_df = app_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    else:
        raise FileNotFoundError(f"Missing {bureau_path}. Please download it from Kaggle.")
    
    # 2. Process and Merge Installments Data
    if os.path.exists(installments_path):
        inst_agg = process_installments(installments_path)
        app_df = app_df.merge(inst_agg, on='SK_ID_CURR', how='left')
    else:
        raise FileNotFoundError(f"Missing {installments_path}. Please download it from Kaggle.")

    # 3. Domain Ratios (The High-Value Features)
    print("Calculating domain-specific financial ratios...")
    # Fill NA for income to prevent critical division by zero errors
    app_df['AMT_INCOME_TOTAL'] = app_df['AMT_INCOME_TOTAL'].fillna(1) 
    
    app_df['ANNUITY_INCOME_PERCENT'] = app_df['AMT_ANNUITY'] / app_df['AMT_INCOME_TOTAL']
    app_df['CREDIT_INCOME_PERCENT'] = app_df['AMT_CREDIT'] / app_df['AMT_INCOME_TOTAL']
    
    # Protect against division by zero for age
    app_df['DAYS_BIRTH'] = app_df['DAYS_BIRTH'].replace(0, -1)
    app_df['DAYS_EMPLOYED_PERCENT'] = app_df['DAYS_EMPLOYED'] / app_df['DAYS_BIRTH']

    # 4. Save Finalized Feature Matrix
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    app_df.to_csv(output_path, index=False)
    print(f"✅ Final engineered feature matrix saved to {output_path} with shape {app_df.shape}")

if __name__ == "__main__":
    main()