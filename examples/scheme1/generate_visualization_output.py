from typing import List
import os
import json
import yaml
import numpy as np
import spikeinterface as si
import figurl as fg
import sortingview.views as vv
from mountainsort5.core.extract_snippets import extract_snippets
from helpers.create_autocorrelograms_view import create_autocorrelograms_view
from helpers.compute_correlogram_data import compute_correlogram_data


def generate_visualization_output(*, rec, recording_preprocessed: si.BaseRecording, sorting: si.BaseSorting, sorting_true: si.BaseSorting):
    os.environ['KACHERY_STORE_FILE_DIR'] = f'output/{rec.recording_name}'
    os.environ['KACHERY_STORE_FILE_PREFIX'] = '$dir'

    if not os.path.exists('output'):
        os.mkdir('output')
    output_dir = os.environ['KACHERY_STORE_FILE_DIR']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(f'{output_dir}/recording'):
        print('Saving preprocessed recording')
        recording_preprocessed.save(folder=f'{output_dir}/recording', format='binary')

    units_dict = {}
    units_dict['true'] = sorting_true.get_unit_spike_train(sorting_true.unit_ids[0], segment_index=0).astype(np.int32)
    for unit_id in sorting.unit_ids:
        units_dict[str(unit_id)] = sorting.get_unit_spike_train(unit_id, segment_index=0).astype(np.int32)
    try:
        # depends on version of SI
        sorting_with_true = si.NumpySorting.from_unit_dict([units_dict], sampling_frequency=sorting.sampling_frequency) # type: ignore
    except: # noqa
        sorting_with_true = si.NumpySorting.from_dict([units_dict], sampling_frequency=sorting.sampling_frequency) # type: ignore

    print('Loading traces')
    traces: np.ndarray = recording_preprocessed.get_traces()
    channel_locations = recording_preprocessed.get_channel_locations()

    unit_ids = sorting_with_true.unit_ids
    channel_ids = recording_preprocessed.channel_ids
    K = len(unit_ids)
    M = len(recording_preprocessed.channel_ids)
    T1 = 20
    T2 = 20
    T = T1 + T2
    print('Compute templates')
    templates = np.zeros((K, T, M), dtype=np.float32)
    for i in range(K):
        unit_id = unit_ids[i]
        times1: np.ndarray = sorting_with_true.get_unit_spike_train(unit_id, segment_index=0)
        snippets1 = extract_snippets(traces, channel_locations=None, mask_radius=None, times=times1, channel_indices=None, T1=T1, T2=T2)
        templates[i] = np.median(snippets1, axis=0)
    peak_channels = {
        str(unit_ids[i]): channel_ids[np.argmin(np.min(templates[i], axis=0))]
        for i in range(K)
    }

    sorting_data = {
        'samplingFrequency': sorting_with_true.get_sampling_frequency(),
        'units': [
            {
                'unitId': f'{unit_id}',
                'peakChannelId': peak_channels[str(unit_id)],
                'spikeTrain': sorting_with_true.get_unit_spike_train(unit_id).astype(np.int32)
            }
            for unit_id in sorting_with_true.unit_ids
            if len(sorting_with_true.get_unit_spike_train(unit_id)) > 0
        ]
    }
    with open(f'{output_dir}/sorting.json', 'w') as f:
        json.dump(fg.serialize_data(sorting_data), f)

    v_et = vv.EphysTraces(
        format='spikeinterface.binary',
        uri='$dir/recording',
        sorting_uri='$dir/sorting.json'
    )

    # v_et_2 = vv.EphysTraces(
    #     format='spikeinterface.binary',
    #     uri='$dir/generated/recording',
    #     sorting_uri=f'$dir/generated/test_mountainsort_sorting.json'
    # )

    # auto-correlograms
    print('Auto correlograms')
    v_ac = create_autocorrelograms_view(sorting=sorting_with_true)

    adjacency_radius = 100
    adjacency = {}
    for m in range(M):
        adjacency[str(channel_ids[m])] = []
        for m2 in range(M):
            dist0 = np.sqrt(np.sum((channel_locations[m] - channel_locations[m2]) ** 2))
            if dist0 <= adjacency_radius:
                adjacency[str(channel_ids[m])].append(str(channel_ids[m2]))

    # cross-correlograms
    print('Cross correlograms')
    cross_correlogram_items: List[vv.CrossCorrelogramItem] = []
    for unit_id1 in sorting_with_true.unit_ids:
        for unit_id2 in sorting_with_true.unit_ids:
            if str(peak_channels[str(unit_id1)]) in adjacency[str(peak_channels[str(unit_id2)])]:
                a = compute_correlogram_data(sorting=sorting_with_true, unit_id1=unit_id1, unit_id2=unit_id2, window_size_msec=80, bin_size_msec=1)
                bin_edges_sec = a['bin_edges_sec']
                bin_counts = a['bin_counts']
                cross_correlogram_items.append(
                    vv.CrossCorrelogramItem(
                        unit_id1=str(unit_id1),
                        unit_id2=str(unit_id2),
                        bin_edges_sec=bin_edges_sec,
                        bin_counts=bin_counts
                    )
                )
    v_cc = vv.CrossCorrelograms(
        cross_correlograms=cross_correlogram_items,
        hide_unit_selector=True
    )

    # units table
    print('Units table')
    v_ut = vv.UnitsTable(
        columns=[
        ],
        rows=[
            vv.UnitsTableRow(str(unit_id), {
            })
            for unit_id in sorting_with_true.get_unit_ids()
        ]
    )

    view = vv.Box(
        direction='horizontal',
        items=[
            vv.LayoutItem(v_ut, stretch=0, min_size=150, max_size=150),
            vv.LayoutItem(v_ac, stretch=0, min_size=400, max_size=400),
            vv.LayoutItem(
                vv.Splitter(
                    direction='horizontal',
                    item1=vv.LayoutItem(v_cc, stretch=1),
                    item2=vv.LayoutItem(
                        vv.TabLayout(
                            items=[
                                vv.TabLayoutItem(label='preprocessed', view=v_et),
                                # vv.TabLayoutItem(label='full', view=v_et_2)
                            ]
                        ),
                        stretch=1
                    )
                ), stretch=1
            )
        ]
    )

    dd = view.url_dict(label=f'{rec.recording_name}')
    with open(f'{output_dir}/view.yaml', 'w') as f:
        yaml.dump(dd, f)
