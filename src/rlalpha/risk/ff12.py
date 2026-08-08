from __future__ import annotations

import numpy as np

FF12_NAMES = ("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")


def _in(sic: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(low <= sic <= high for low, high in ranges)


def ff12_industry(sic: float | int | None) -> str:
    if sic is None or not np.isfinite(float(sic)):
        return "Other"
    value = int(sic)
    groups = {
        "NoDur": ((100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)),
        "Durbl": ((2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)),
        "Manuf": ((2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)),
        "Enrgy": ((1200, 1399), (2900, 2999)),
        "Chems": ((2800, 2829), (2840, 2899)),
        "BusEq": ((3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)),
        "Telcm": ((4800, 4899),),
        "Utils": ((4900, 4949),),
        "Shops": ((5000, 5999), (7200, 7299), (7600, 7699)),
        "Hlth": ((2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)),
        "Money": ((6000, 6999),),
    }
    for name, ranges in groups.items():
        if _in(value, ranges):
            return name
    return "Other"

