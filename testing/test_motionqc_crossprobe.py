import numpy as np
import pandas as pd

from motionqc.crossprobe import fit_clock_map, map_intervals


def test_clock_map_recovers_affine_mapping_and_intervals():
    source=np.arange(1000,dtype=float);destination=1.000002*source-.003
    mapping=fit_clock_map(source,destination,source="a",destination="b")
    assert abs(mapping.slope-1.000002)<1e-12
    assert abs(mapping.intercept_s+.003)<1e-12
    assert mapping.residual_max_abs_s<1e-10
    frame=pd.DataFrame({"start_s":[1.],"end_s":[2.]})
    mapped=map_intervals(frame,mapping)
    assert np.allclose(mapped[["start_s","end_s"]],[[.997002,1.997004]])
