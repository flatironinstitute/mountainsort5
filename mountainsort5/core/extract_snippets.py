from typing import Union
import numpy as np


def extract_snippets(
    traces: np.ndarray, *,
    channel_locations: Union[np.ndarray, None],
    mask_radius: Union[float, None],
    times: np.array,
    channels: Union[np.array, None],
    T1: int,
    T2: int
) -> np.ndarray:
    M = traces.shape[1]
    L = len(times)

    if mask_radius is not None:
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
            channel_inds = adjacency[channels[j]]
            snippets[j][:, channel_inds] = traces[t1:t2, channel_inds]
        else:
            snippets[j] = traces[t1:t2]
    return snippets