import csv
from collections import OrderedDict
import numpy as np
import json

class Logger:
    def __init__(self):
        self.rows = []

    def log(self, **kw):
        row = OrderedDict()
        for k,v in kw.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                row[k] = np.asarray(v).tolist()
            elif isinstance(v, dict):
                row[k] = v  
            else:
                row[k] = v
        self.rows.append(row)

    def to_csv(self, path):    
        if not self.rows:
            return
        keys = OrderedDict()
        for r in self.rows:
            for k in r.keys():
                keys[k] = True
        keys = list(keys.keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                rr = {}
                for k in keys:
                    v = r.get(k, "")
                    if isinstance(v, (list, dict)):
                        rr[k] = json.dumps(v)
                    else:
                        rr[k] = v
                w.writerow(rr)