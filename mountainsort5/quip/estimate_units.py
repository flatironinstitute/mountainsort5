from typing import Union, List
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
from dataclasses import dataclass
from ..schemes.Scheme1SortingParameters import Scheme1SortingParameters
from ..schemes.sorting_scheme2 import get_time_chunks
from ..core.get_block_recording_for_scheme3 import get_block_recording_for_scheme3
from ..schemes.sorting_scheme1 import sorting_scheme1
from ..schemes.sorting_scheme2 import TimeChunk


@dataclass
class EstimateUnitsParameters:
    block_sorting_parameters: Scheme1SortingParameters = Scheme1SortingParameters()
    avg_num_channels_per_neighborhood: Union[int, None] = 7
    block_duration_sec: float = 300
    max_num_blocks: Union[int, None] = 10

    def check_valid(
        self,
        *,
        M: int,
        N: int,
        sampling_frequency: float,
        channel_locations: np.ndarray,
    ):
        """Internal function for checking validity of parameters"""
        self.block_sorting_parameters.check_valid(
            M=M,
            N=N,
            sampling_frequency=sampling_frequency,
            channel_locations=channel_locations,
        )
        assert self.block_duration_sec > 0


@dataclass
class EstimateUnitsUnit:
    unit_id: Union[int, str]
    num_spikes: int
    peak_channel_id: Union[int, str]
    snr: float

    def to_dict(self):
        return {
            "unit_id": self.unit_id,
            "num_spikes": self.num_spikes,
            "peak_channel_id": self.peak_channel_id,
            "snr": self.snr,
        }


@dataclass
class EstimateUnitsBlock:
    start_time_sec: float
    end_time_sec: float
    units: List[EstimateUnitsUnit]

    def to_dict(self):
        return {
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
            "units": [u.to_dict() for u in self.units],
        }


@dataclass
class EstimateUnitsOutput:
    blocks: List[EstimateUnitsBlock]

    def to_dict(self):
        return {
            "blocks": [b.to_dict() for b in self.blocks],
        }

    def __repr__(self):
        ret = f"EstimateUnitsOutput(blocks={len(self.blocks)})"
        for i, block in enumerate(self.blocks):
            ret += "\n"
            ret += f"  block {i}: {len(block.units)} units"
            for j, unit in enumerate(block.units):
                ret += "\n"
                ret += f"    unit {j}: firing rate {unit.num_spikes / (block.end_time_sec - block.start_time_sec):.2f} Hz, peak channel {unit.peak_channel_id}, snr {unit.snr:.2f}"
        return ret


def estimate_units(
    recording: si.BaseRecording,
    estimate_units_parameters: EstimateUnitsParameters = EstimateUnitsParameters(),
) -> EstimateUnitsOutput:
    ###################################################################
    # Handle multi-segment recordings
    if recording.get_num_segments() > 1:
        raise Exception("Multi-segment recordings not supported for estimate_units")
    ###################################################################

    output = EstimateUnitsOutput(blocks=[])

    M = recording.get_num_channels()
    N = recording.get_num_frames()
    sampling_frequency = recording.sampling_frequency
    channel_locations = recording.get_channel_locations()

    if estimate_units_parameters.avg_num_channels_per_neighborhood is not None:
        radius = _auto_detect_channel_neighborhood_radius(
            channel_locations,
            estimate_units_parameters.avg_num_channels_per_neighborhood,
        )
        if (
            estimate_units_parameters.block_sorting_parameters.snippet_mask_radius
            is None
        ):
            estimate_units_parameters.block_sorting_parameters.snippet_mask_radius = (
                radius
            )
            # raise Exception('TEMP')
        else:
            raise Exception(
                "Cannot specify both avg_num_channels_per_neighborhood and snippet_mask_radius"
            )
        if (
            estimate_units_parameters.block_sorting_parameters.detect_channel_radius
            is None
        ):
            estimate_units_parameters.block_sorting_parameters.detect_channel_radius = (
                radius
            )
        else:
            raise Exception(
                "Cannot specify both avg_num_channels_per_neighborhood and detect_channel_radius"
            )

    if estimate_units_parameters.block_sorting_parameters.skip_alignment is None:
        # cuts the computation time in half
        estimate_units_parameters.block_sorting_parameters.skip_alignment = True

    estimate_units_parameters.check_valid(
        M=M,
        N=N,
        sampling_frequency=sampling_frequency,
        channel_locations=channel_locations,
    )

    block_size = int(
        estimate_units_parameters.block_duration_sec * sampling_frequency
    )  # size of chunks in samples
    blocks: List[TimeChunk] = get_time_chunks(
        np.int64(recording.get_num_samples()),
        chunk_size=np.int32(block_size),
        padding=np.int32(1000),
        max_num_blocks=estimate_units_parameters.max_num_blocks,
    )

    for i, block in enumerate(blocks):
        print("")
        print("=============================================")
        print(f"Processing block {i + 1} of {len(blocks)}...")
        subrecording = get_block_recording_for_scheme3(
            recording=recording,
            start_frame=int(block.start) - int(block.padding_left),
            end_frame=int(block.end) + int(block.padding_right),
        )
        subrecording_filtered = spre.bandpass_filter(
            subrecording, freq_min=300, freq_max=6000, dtype=np.float32
        )
        subrecording_preprocessed = spre.whiten(subrecording_filtered)
        aa = sorting_scheme1(
            subrecording_preprocessed,
            sorting_parameters=estimate_units_parameters.block_sorting_parameters,
            return_extra_output=True,
        )
        assert isinstance(aa, tuple)
        subsorting, extra_output = aa
        assert isinstance(subsorting, si.NumpySorting)
        labels0 = extra_output.labels
        templates0 = extra_output.templates
        peak_channel_indices0 = extra_output.peak_channel_indices
        estimate_units_block = EstimateUnitsBlock(
            start_time_sec=float(block.start / sampling_frequency),
            end_time_sec=float(block.end / sampling_frequency),
            units=[],
        )
        output.blocks.append(estimate_units_block)
        for j in range(1, np.max(labels0) + 1):
            inds0 = np.where(labels0 == j)[0]
            template0 = templates0[j - 1, :, :]
            peak_channel_ind0 = peak_channel_indices0[j - 1]
            snr0 = np.max(np.abs(template0))
            if len(inds0) > 0:
                estimate_units_block.units.append(
                    EstimateUnitsUnit(
                        unit_id=j,
                        num_spikes=len(inds0),
                        peak_channel_id=recording.get_channel_ids()[peak_channel_ind0],
                        snr=snr0
                    )
                )

    return output


def _auto_detect_channel_neighborhood_radius(
    channel_locations: np.ndarray, avg_num_channels_per_neighborhood: int
):
    M = channel_locations.shape[0]
    dist_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            dist_matrix[i, j] = np.linalg.norm(
                channel_locations[i, :] - channel_locations[j, :]
            )
    unique_distances = np.sort(np.unique(dist_matrix))
    avg_nbhd_sizes = []
    for i in range(len(unique_distances)):
        avg_nbhd_sizes.append(
            np.mean(np.sum(dist_matrix <= unique_distances[i], axis=1))
        )
    avg_nbhd_sizes = np.array(avg_nbhd_sizes)
    if np.min(avg_nbhd_sizes) > avg_num_channels_per_neighborhood:
        return float(1)
    if np.max(avg_nbhd_sizes) < avg_num_channels_per_neighborhood:
        return float(np.max(unique_distances))
    ind = np.where(avg_nbhd_sizes >= avg_num_channels_per_neighborhood)[0][0]
    print(
        "------------------- a",
        unique_distances,
        avg_nbhd_sizes,
        ind,
        unique_distances[ind],
    )
    return float(unique_distances[ind])
