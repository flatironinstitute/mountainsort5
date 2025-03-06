import numpy as np
import numpy.typing as npt


def remove_duplicate_events(times: npt.NDArray, labels: npt.NDArray, *, tol: int) -> npt.NDArray:
    new_labels = np.array(labels)
    unit_ids = np.unique(new_labels)
    for unit_id in unit_ids:
        unit_inds = np.nonzero(new_labels == unit_id)[0]
        unit_times = times[unit_inds]
        inds_duplicate = find_duplicate_times(unit_times, tol=tol)
        new_labels[unit_inds[inds_duplicate]] = 0
    inds_nonzero = np.nonzero(new_labels)[0]
    return inds_nonzero

def find_duplicate_times(times: npt.NDArray, *, tol: int) -> npt.NDArray:
    ret: list[np.int32] = []
    deleted = np.zeros((len(times),), dtype=np.int16)
    for i1 in range(len(times)):
        if not deleted[i1]:
            i2 = i1 + 1
            while i2 < len(times) and times[i2] <= times[i1] + tol:
                ret.append(np.int32(i2))
                deleted[i2] = True
                i2 += 1
    return np.array(ret, dtype=np.int32)
