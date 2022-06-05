import pandas as pd

def load_dataset(pth):
    if pth.endswith('.csv'):
        pd.read_csv(pth)