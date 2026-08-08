# Data dictionary

Schema discovery resolved six Parquet files in `/data/sunyuxiang/rl_alpha`: daily stock (3,131,836 rows), historical membership (911), market daily (4,529), CCM links (907), annual Compustat (14,531), and delistings (330). Discovery is based on required columns, not filenames.

The delivered CRSP membership file encodes active intervals with `MbrFlg=NORM` (all 911 rows), not `Y`. The loader accepts the documented `Y` and observed `NORM` encodings, while still requiring inclusive start/end interval membership.

`adj_open/high/low/close = DlyOpen/High/Low/Close / DlyCumFacPr`; `adj_volume = DlyVol * DlyCumFacShr`. Non-positive or missing factors produce NaN. `DlyRet` is authoritative total return. `DelRet` is used only when a CIZ row is flagged as a delisting and `DlyRet` is missing.

CRSP `DlyCap` and Compustat accounting fields are treated as thousand USD and million USD respectively only after the implemented `DlyCap / (abs(DlyClose) * ShrOut)` QA identity is checked. Annual fundamentals become available six calendar months after `datadate` and expire after 18 months.
