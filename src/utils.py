'''
utility functions for data inspection and helper tasks
'''
from src.data_loader import load_raw_data

def inspect_columns(path):
    '''
    loads data from the given path and prints available columns for params.yaml
    '''
    try:
        df = load_raw_data(path)
        
        print("\n" + "="*50)
        print(f"inspection for: {path}")
        print("="*50)
        
        print("\nfound columns (copy-paste these to model_cols in params.yaml):")
        for col in df.columns:
            print(f"  - '{col}'")
            
        print("\ndata types:")
        print(df.dtypes)
        
        print("\npreview (first 3 rows):")
        print(df.head(3))
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"error during inspection: {e}")

if __name__ == "__main__":
    # this allows running the script standalone for quick checks
    import argparse
    parser = argparse.ArgumentParser(description="data inspector utility")
    parser.add_argument("--path", type=str, required=True, help="path to the file or directory to inspect")
    
    args = parser.parse_args()
    inspect_columns(args.path)