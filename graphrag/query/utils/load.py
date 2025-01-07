from typing import Any

import numpy as np
import pandas as pd


def load_parquet(path) -> list[dict[str, Any]]:
    records = pd.read_parquet(path, engine="pyarrow").to_dict(orient="records")
    for record in records:
        for k, v in record.items():
            if isinstance(v, (np.ndarray, pd.Series)):
                record[k] = v.tolist()
    return records