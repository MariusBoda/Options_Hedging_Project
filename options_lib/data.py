import pandas as pd
    
def data_load(file_path):
    df = pd.read_feather(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.index.name = None 
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    return df


