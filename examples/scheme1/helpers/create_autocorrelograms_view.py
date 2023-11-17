from typing import List
import spikeinterface as si
import sortingview.views as vv
from helpers.compute_correlogram_data import compute_correlogram_data


def create_autocorrelograms_view(*, sorting: si.BaseSorting, unit_id_prefix: str = ''):
    autocorrelogram_items: List[vv.AutocorrelogramItem] = []
    for unit_id in sorting.get_unit_ids():
        a = compute_correlogram_data(sorting=sorting, unit_id1=unit_id, unit_id2=None, window_size_msec=80, bin_size_msec=1)
        bin_edges_sec = a['bin_edges_sec']
        bin_counts = a['bin_counts']
        autocorrelogram_items.append(
            vv.AutocorrelogramItem(
                unit_id=f'{unit_id_prefix}{unit_id}',
                bin_edges_sec=bin_edges_sec,
                bin_counts=bin_counts
            )
        )
    view = vv.Autocorrelograms(
        autocorrelograms=autocorrelogram_items
    )
    return view
