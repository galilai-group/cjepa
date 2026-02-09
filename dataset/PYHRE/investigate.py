import hickle as hkl
import os

def load_hkl(path):
    data = hkl.load(path)
    return data

basedir = "dataset/PYHRE"

data_samples = os.listdir(basedir)

for sample in data_samples: 
    if sample.endswith('.hkl'):
        path = os.path.join(basedir, sample)
        data = load_hkl(path)
        print(f"Loaded {sample}")
        print(data.shape if hasattr(data, 'shape') else type(data))
        print(data)
