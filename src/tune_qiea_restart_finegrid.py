"""
Follow-up to tune_qiea_restart.py: the first screen's winning restart_patience (20)
was the SHORTEST value on that grid -- an edge-of-grid result, same situation as
section 8c's theta_min range check. This re-screens patience at and below 20 to
confirm 20 isn't itself an artificially-truncated optimum, holding restart_fraction
at the two values (0.3, 0.5) that looked best in the first screen.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_qiea_restart as t  # noqa: E402

t.PATIENCE_GRID = [5, 10, 15, 20, 30, 40]
t.FRACTION_GRID = [0.3, 0.5]

if __name__ == "__main__":
    import pandas as pd

    all_rows = []
    for inst in t.INSTANCES:
        df = t.run_instance(inst)
        print(df.to_string(index=False))
        print()
        all_rows.append(df)
    pd.concat(all_rows).to_csv(
        Path(__file__).resolve().parent.parent / "results" / "tune_qiea_restart_finegrid.csv", index=False
    )
