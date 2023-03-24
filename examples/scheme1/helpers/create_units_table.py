from typing import List
import sortingview.views as vv
import spikeinterface as si


def create_units_table(*, sorting: si.BaseSorting):
    columns: List[vv.UnitsTableColumn] = []
    rows: List[vv.UnitsTableRow] = []
    for unit_id in sorting.get_unit_ids():
        rows.append(
            vv.UnitsTableRow(
                unit_id=unit_id,
                values={
                    'unitId': unit_id
                }
            )
        )
    view = vv.UnitsTable(
        columns=columns,
        rows=rows
    )
    return view
