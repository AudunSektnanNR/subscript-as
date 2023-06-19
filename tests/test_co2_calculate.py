import itertools
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage
import shapely.geometry
import xtgeo

from subscript.co2containment.calculate import (
    calculate_co2_containment,
    # calculate_co2_mass,
    # SourceData,
)

from subscript.co2containment.co2_calculation import (
    SourceData,
    _calculate_co2_data_from_source_data,
    CalculationType,
    # Co2Data,
)

from subscript.co2containment.co2_containment import calculate_from_co2_data


def _random_prop(dims, rng, low, high):
    white = rng.normal(size=dims)
    smooth = scipy.ndimage.gaussian_filter(white, max(dims) / 10)
    values = smooth - np.min(smooth)
    values /= np.max(values)
    values *= high
    values += low
    return values.flatten()


def _xy_and_volume(grid: xtgeo.Grid):
    xyz = grid.get_xyz()
    vol = grid.get_bulk_volume().values1d.compressed()
    return xyz[0].values1d.compressed(), xyz[1].values1d.compressed(), vol


@pytest.fixture
def dummy_co2_grid():
    dims = (11, 13, 7)
    return xtgeo.create_box_grid(dims)


@pytest.fixture
def dummy_co2_masses(dummy_co2_grid):
    dims = dummy_co2_grid.dimensions
    nt = 10
    rng = np.random.RandomState(123)
    x, y, vol = _xy_and_volume(dummy_co2_grid)
    dates = [str(2020 + i) for i in range(nt)]
    source_data = SourceData(
        x,
        y,
        PORV={date: _random_prop(dims, rng, 0.1, 0.3) for date in dates},
        VOL=vol,
        DATES=dates,
        SWAT={date: _random_prop(dims, rng, 0.05, 0.6) for date in dates},
        DWAT={date: _random_prop(dims, rng, 950, 1050) for date in dates},
        SGAS={date: _random_prop(dims, rng, 0.05, 0.6) for date in dates},
        DGAS={date: _random_prop(dims, rng, 700, 850) for date in dates},
        AMFG={date: _random_prop(dims, rng, 0.001, 0.01) for date in dates},
        YMFG={date: _random_prop(dims, rng, 0.001, 0.01) for date in dates}
    )
    return _calculate_co2_data_from_source_data(source_data, CalculationType.mass)


def _calc_and_compare(poly, masses, poly_hazardous=None):
    totals = {m.date: np.sum(m.total_mass()) for m in masses.data_list}
    contained = calculate_from_co2_data(
        co2_data=masses,
        containment_polygon=poly,
        hazardous_polygon=poly_hazardous,
        compact=False,
        calc_type_input="mass"
    )
    difference = np.sum([x-y for x,y in zip(contained.total.values, list(totals.values()))])
    assert(difference == pytest.approx(0.0, abs=1e-8))
    return contained


def test_single_poly_co2_containment(dummy_co2_masses):
    assert len(dummy_co2_masses.data_list) == 10
    poly = shapely.geometry.Polygon([
        [7.1, 7.0],
        [9.1, 9.0],
        [7.1, 11.0],
        [5.1, 9.0],
        [7.1, 7.0],
    ])
    contained = _calc_and_compare(poly, dummy_co2_masses)
    assert(contained.gas_contained.values[-1] == pytest.approx(90.262207))
    assert(contained.aqueous_contained.values[-1] == pytest.approx(172.72921760648467))
    assert(contained.gas_hazardous.values[-1] == pytest.approx(0.0))
    assert(contained.aqueous_hazardous.values[-1] == pytest.approx(0.0))


def test_multi_poly_co2_containment(dummy_co2_masses):
    poly = shapely.geometry.MultiPolygon([
        shapely.geometry.Polygon([
            [7.1, 7.0],
            [9.1, 9.0],
            [7.1, 11.0],
            [5.1, 9.0],
            [7.1, 7.0],
        ]),
        shapely.geometry.Polygon([
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 3.0],
            [1.0, 3.0],
            [1.0, 1.0],
        ]),
    ])
    contained = _calc_and_compare(poly, dummy_co2_masses)
    assert(contained.gas_contained.values[-1] == pytest.approx(123.70267352027123))
    assert(contained.aqueous_contained.values[-1] == pytest.approx(252.79970312163525))
    assert(contained.gas_hazardous.values[-1] == pytest.approx(0.0))
    assert(contained.aqueous_hazardous.values[-1] == pytest.approx(0.0))


def test_hazardous_poly_co2_containment(dummy_co2_masses):
    assert len(dummy_co2_masses.data_list) == 10
    poly = shapely.geometry.Polygon([
        [7.1, 7.0],
        [9.1, 9.0],
        [7.1, 11.0],
        [5.1, 9.0],
        [7.1, 7.0],
    ])
    poly_hazardous = shapely.geometry.Polygon([
        [9.1, 9.0],
        [9.1, 11.0],
        [7.1, 11.0],
        [9.1, 9.0],
    ])
    contained = _calc_and_compare(poly, dummy_co2_masses, poly_hazardous)
    assert(contained.gas_contained.values[-1] == pytest.approx(90.262207))
    assert(contained.aqueous_contained.values[-1] == pytest.approx(172.72921760648467))
    assert(contained.gas_hazardous.values[-1] == pytest.approx(12.687891108274542))
    assert(contained.aqueous_hazardous.values[-1] == pytest.approx(20.33893251315071))


def test_reek_grid():
    reek_gridfile = (
        Path(__file__).absolute().parent
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0.EGRID"
    )
    reek_poly = shapely.geometry.Polygon([
        [461339, 5932377],
        [461339 + 1000, 5932377],
        [461339 + 1000, 5932377 + 1000],
        [461339, 5932377 + 1000],
    ])
    grid = xtgeo.grid_from_file(reek_gridfile)
    poro = xtgeo.gridproperty_from_file(
        reek_gridfile.with_suffix(".INIT"), name="PORO", grid=grid
    ).values1d.compressed()
    x, y, vol = _xy_and_volume(grid)
    source_data = SourceData(
        x,
        y,
        np.ones_like(poro) * 0.1,
        vol,
        ["2042"],
        [np.ones_like(poro) * 0.1],
        [np.ones_like(poro) * 1000.0],
        [np.ones_like(poro) * 0.1],
        [np.ones_like(poro) * 100.0],
        [np.ones_like(poro) * 0.1],
        [np.ones_like(poro) * 0.1],
    )
    mass = calculate_co2_mass(source_data)
    table = calculate_co2_containment(
        source_data.x, source_data.y, mass, reek_poly
    )
    assert sum(t.amount_kg for t in table if t.inside_boundary) == pytest.approx(89498504)
