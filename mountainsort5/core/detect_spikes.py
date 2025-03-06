from typing import Tuple, Union
import numpy as np
import numpy.typing as npt


def detect_spikes(
    traces: npt.NDArray[np.float32], *,
    channel_locations: npt.NDArray[np.float32],
    time_radius: int,
    channel_radius: Union[float, None],
    detect_threshold: float,
    detect_sign: int,
    margin_left: int,
    margin_right: int,
    verbose: bool
) -> Tuple[npt.NDArray, npt.NDArray]:
    N = traces.shape[0]
    M = traces.shape[1]

    if detect_sign > 0:
        # todo: figure out how to avoid making a copy
        traces = -traces # pragma: no cover
    elif detect_sign == 0:
        # todo: figure out how to avoid making a copy
        traces = -np.abs(traces) # pragma: no cover

    adjacency = []
    for m in range(M):
        adjacency.append([])
        for m2 in range(M):
            dist0 = np.sqrt(np.sum((channel_locations[m] - channel_locations[m2]) ** 2))
            if (channel_radius is None) or (dist0 <= channel_radius):
                adjacency[m].append(m2)
    print('')
    print(f'Adjacency for detect spikes with channel radius {channel_radius}')
    print(adjacency)
    print('')

    inds1, inds2 = np.nonzero(traces <= -detect_threshold)

    candidate_times = [[] for m in range(M)]
    candidate_values = [[] for m in range(M)]
    for i in range(len(inds1)):
        if inds1[i] >= margin_left and inds1[i] < N - margin_right:
            candidate_times[inds2[i]].append(inds1[i])
            candidate_values[inds2[i]].append(traces[inds1[i], inds2[i]])

    times = []
    channel_indices = []
    for m in range(M):
        nbhd = adjacency[m]
        if verbose:
            print(f'm = {m} (nbhd size: {len(nbhd)})')
        indices = [0 for j in range(len(nbhd))]
        for i in range(len(candidate_times[m])):
            t = candidate_times[m][i]
            v = candidate_values[m][i]
            okay = True
            for j in range(len(nbhd)):
                if not okay:
                    break
                tt = candidate_times[nbhd[j]]
                vv = candidate_values[nbhd[j]]
                ii = indices[j]
                while ii < len(tt) and tt[ii] < t - time_radius:
                    ii += 1
                indices[j] = ii # advance
                jj = ii
                while jj < len(tt) and tt[jj] <= t + time_radius:
                    if vv[jj] < v:
                        okay = False
                        break
                    jj += 1
            if okay:
                times.append(t)
                channel_indices.append(m)

    times = np.array(times, dtype=np.int32)
    channel_indices = np.array(channel_indices, dtype=np.int32)
    inds = np.argsort(times)
    times = times[inds]
    channel_indices = channel_indices[inds]
    return times, channel_indices
