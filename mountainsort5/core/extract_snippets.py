from typing import Union, List
import numpy as np
import numpy.typing as npt


def extract_snippets(
    traces: npt.NDArray[np.float32], *,
    channel_locations: Union[npt.NDArray[np.float32], None],
    mask_radius: Union[float, None],
    times: npt.NDArray,
    channel_indices: Union[npt.NDArray, None],
    T1: int,
    T2: int
) -> npt.NDArray[np.float32]:
    M = traces.shape[1]
    L = len(times)

    if mask_radius is not None:
        assert channel_locations is not None
        assert channel_indices is not None
        adjacency = []
        for m in range(M):
            adjacency.append([])
            for m2 in range(M):
                dist0 = np.sqrt(np.sum((channel_locations[m] - channel_locations[m2]) ** 2))
                if dist0 <= mask_radius:
                    adjacency[m].append(m2)
    else:
        adjacency = None

    snippets = np.zeros((L, T1 + T2, M), dtype=np.float32)
    for j in range(L):
        t1 = times[j] - T1
        t2 = times[j] + T2
        if adjacency is not None:
            assert channel_indices is not None
            channel_inds = adjacency[channel_indices[j]]
            snippets[j][:, channel_inds] = traces[t1:t2, channel_inds]
        else:
            snippets[j] = traces[t1:t2]
    return snippets

def extract_snippets_in_channel_neighborhood(
    traces: npt.NDArray[np.float32], *,
    times: npt.NDArray,
    neighborhood: Union[List[int], None],
    T1: int,
    T2: int
) -> np.ndarray:
    L = len(times)

    if neighborhood is None:
        neighborhood = list(range(traces.shape[1]))

    snippets = np.zeros((L, T1 + T2, len(neighborhood)), dtype=np.float32)
    for j in range(L):
        t1 = times[j] - T1
        t2 = times[j] + T2
        snippets[j] = traces[t1:t2][:, neighborhood]

    return snippets
