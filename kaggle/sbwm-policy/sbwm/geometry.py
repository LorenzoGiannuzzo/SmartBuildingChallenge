"""Static per-zone geometry features computed from the policy metadata.

The floor plan and zone map are part of ``building_meta_data``. Rooms are
recovered with the simulator's own preprocessing, so room ids match the ids in
``ahu_zone_assignments``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

try:
    from smart_buildings.smart_control.simulator import building_utils
except ImportError as err:
    #Lorenzo Giannuzzo: the floor-plan preprocessing only needs gin for a
    #decorator elsewhere in the module, so a minimal stand-in is enough where
    #gin-config is not installed (for example a bare Kaggle notebook image).
    if "gin" not in str(err):
        raise
    import sys
    import types

    _gin = types.ModuleType("gin")
    _gin.configurable = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    sys.modules["gin"] = _gin
    from smart_buildings.smart_control.simulator import building_utils
from smart_buildings.smart_control.simulator import constants

ZONE_STATIC_FIELDS = (
    "log_area",
    "exterior_fraction",
    "exterior_per_sqrt_area",
    "floor_position",
    "is_ground_floor",
    "is_top_floor",
    "zones_on_ahu",
)


def zone_static_features(meta, schema) -> np.ndarray:
    """Return a (V, len(ZONE_STATIC_FIELDS)) float32 array."""
    floor_plan = meta.get("floor_plan")
    zone_map = meta.get("zone_map")
    v = schema.num_zones
    out = np.zeros((v, len(ZONE_STATIC_FIELDS)), np.float32)
    counts = np.bincount(schema.zone_ahu, minlength=schema.num_ahus)
    out[:, 6] = counts[schema.zone_ahu] / 10.0
    if floor_plan is None or zone_map is None:
        return out

    room_dict, exterior_walls, _, exterior_space, _ = (
        building_utils.construct_building_data_types(floor_plan, zone_map)
    )
    is_ext_wall = exterior_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
    near_ext = ndimage.binary_dilation(is_ext_wall, iterations=2)

    #Lorenzo Giannuzzo: each floor is a separate connected footprint in the 2D
    #plan; floors are ordered top-to-bottom, left-to-right in the image.
    footprint = exterior_space != constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION
    labels, n_floors = ndimage.label(footprint)
    if n_floors > 1:
        centers = ndimage.center_of_mass(footprint, labels, range(1, n_floors + 1))
        order = np.lexsort(([c[1] for c in centers], [round(c[0] / 50) for c in centers]))
        rank = {int(lab) + 1: r for r, lab in enumerate(order)}
    else:
        rank = {1: 0}

    for z, room in enumerate(schema.zone_rooms):
        coords = room_dict.get(room)
        if not coords:
            continue
        rows, cols = np.asarray(coords).T
        area = float(len(coords))
        mask = np.zeros(is_ext_wall.shape, bool)
        mask[rows, cols] = True
        edge = mask & ~ndimage.binary_erosion(mask)
        ext_cells = float((edge & near_ext).sum())
        floor_label = int(np.bincount(labels[rows, cols]).argmax())
        position = rank.get(floor_label, 0) / max(n_floors - 1, 1)
        out[z, 0] = np.log(area) / 8.0
        out[z, 1] = ext_cells / max(edge.sum(), 1)
        out[z, 2] = ext_cells / np.sqrt(area)
        out[z, 3] = position
        out[z, 4] = float(rank.get(floor_label, 0) == 0)
        out[z, 5] = float(rank.get(floor_label, 0) == n_floors - 1)
    return out
