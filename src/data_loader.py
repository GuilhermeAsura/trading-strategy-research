import pandas as pd
import numpy as np
import os
import glob

def load_raw_data(path):
    # english comments, lowercase, succinct
    if not os.path.exists(path):
        print(f"debug: file not found at {path}")
        return None
    
    encodings = ['utf-8', 'latin-1', 'cp1252']
    df = None
    
    for enc in encodings:
        try:
            # find header: first line with at least 3 separators
            with open(path, 'r', encoding=enc) as f:
                header_idx = 0
                for i in range(15):
                    line = f.readline()
                    if line.count(',') >= 3 or line.count(';') >= 3:
                        header_idx = i
                        break
            
            df = pd.read_csv(path, encoding=enc, sep=None, engine='python', 
                             skiprows=header_idx, on_bad_lines='skip')
            break
        except Exception as e:
            continue
            
    if df is not None:
        # normalize columns
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # dynamic result column detection
        target_cols = ['resultado', 'result_dia', 'lucro', 'p/l', 'diário l.']
        found_col = next((c for c in target_cols if c in df.columns), None)
        
        if found_col:
            # clean currency/accounting strings
            s = df[found_col].astype(str).str.replace('"', '').str.strip()
            is_neg = s.str.contains(r'\(.*\)')
            s = s.str.replace(r'[\(\)]', '', regex=True)
            s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            
            # numeric extraction
            vals = pd.to_numeric(s.str.extract(r'([-+]?\d*\.?\d+)')[0], errors='coerce')
            df['res__operação'] = np.where(is_neg, -vals, vals)
            df['res__operação'] = df['res__operação'].fillna(0)
            
    return df

def create_returns_matrix(path):
    # english comments, lowercase, succinct
    print(f"debug: processing path -> {path}")
    
    # case 1: path is a directory (legacy)
    if os.path.isdir(path):
        all_files = glob.glob(os.path.join(path, "*.csv"))
        print(f"debug: directory mode. found {len(all_files)} files.")
        strategy_columns = {}
        for f in all_files:
            name = os.path.basename(f).replace('.csv', '')
            temp_df = load_raw_data(f)
            if temp_df is not None and 'res__operação' in temp_df.columns:
                date_col = next((c for c in ['data', 'abertura', 'date'] if c in temp_df.columns), None)
                if date_col:
                    temp_df.index = pd.to_datetime(temp_df[date_col], errors='coerce', dayfirst=True)
                    daily = temp_df['res__operação'].resample('D').sum().fillna(0)
                    strategy_columns[name] = daily
            else:
                print(f"debug: skipping {name} - columns not found")
        return pd.DataFrame(strategy_columns).fillna(0)

    # case 2: path is a compiled master file (recommended)
    if os.path.isfile(path):
        print("debug: file mode. parsing master sheet...")
        df = load_raw_data(path)
        if df is None: return pd.DataFrame()
        
        # map master sheet columns
        strat_col = next((c for c in ['arquivo', 'estratégia', 'strategy'] if c in df.columns), None)
        date_col = next((c for c in ['data', 'abertura', 'date'] if c in df.columns), None)
        
        if not strat_col or not date_col:
            print(f"debug: missing required columns. found: {df.columns.tolist()}")
            return pd.DataFrame()
            
        # conversion and pivoting
        df.index = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
        df = df[df.index.notnull()]
        
        # transform long format to wide returns matrix
        returns_matrix = df.pivot_table(
            index=df.index, 
            columns=strat_col, 
            values='res__operação', 
            aggfunc='sum'
        ).resample('D').sum().fillna(0)
        
        print(f"success: matrix created ({returns_matrix.shape[1]} strategies)")
        return returns_matrix

    return pd.DataFrame()