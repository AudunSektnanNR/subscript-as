import dataclasses

import numpy as np
import pytest
import shapely.geometry

from subscript.co2containment.co2_calculation import (
    SourceData,
    _calculate_co2_data_from_source_data,
    CalculationType,
    Co2Data,
)


@pytest.fixture
def simple_cube_grid():
    dims = (13, 17, 19)
    mx, my, mz = np.meshgrid(
        np.linspace(-1, 1, dims[0]),
        np.linspace(-1, 1, dims[1]),
        np.linspace(-1, 1, dims[2]),
        indexing="ij"
    )
    dates = [f"{d}0101" for d in range(2030, 2050)]
    dists = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    gas_saturations = {}
    for i in range(len(dates)):
        gas_saturations[dates[i]] = np.maximum(np.exp(-3 * (dists.flatten() / ((i + 1) / len(dates))) ** 2) - 0.05, 0.0)
    size = np.prod(dims)
    return SourceData(
        mx.flatten(),
        my.flatten(),
        PORV={date: np.ones(size) * 0.3 for date in dates},
        VOL={date: np.ones(size) * (8 / size) for date in dates},
        DATES=dates,
        DWAT={date: np.ones(size) * 1000.0 for date in dates},
        SWAT={date: 1 - gas_saturations[date] for date in gas_saturations},
        SGAS=gas_saturations,
        DGAS={date: np.ones(size) * 100.0 for date in dates},
        AMFG={date: np.ones(size) * 0.02 * gas_saturations[date] for date in gas_saturations},
        YMFG={date: np.ones(size) * 0.99 for date in dates},
    )


@pytest.fixture
def simple_poly():
    return shapely.geometry.Polygon(np.array([
        [-0.45, -0.38],
        [0.41, -0.39],
        [0.33, 0.76],
        [-0.27, 0.75],
        [-0.45, -0.38],
    ]))


def test_simple_cube_grid(simple_cube_grid, simple_poly):
    co2_data = _calculate_co2_data_from_source_data(simple_cube_grid,
                                                    CalculationType.mass)
    assert(len(co2_data.data_list) == len(simple_cube_grid.DATES))
    assert(co2_data.units == "kg")
    assert(co2_data.data_list[-1].date == "20490101")
    assert(co2_data.data_list[-1].gas_phase.sum() == pytest.approx(9585.032869548137))
    assert(co2_data.data_list[-1].aqu_phase.sum() == pytest.approx(2834.956447728449))


def test_zoned_simple_cube_grid(simple_cube_grid, simple_poly):
    rs = np.random.RandomState(123)
    zone = rs.choice([1, 2, 3], size=simple_cube_grid.PORV['20300101'].shape)
    simple_cube_grid.zone = zone
    co2_data = _calculate_co2_data_from_source_data(simple_cube_grid,
                                              CalculationType.mass)
    assert isinstance(co2_data, Co2Data)
    assert(co2_data.data_list[-1].date == "20490101")
    assert(co2_data.data_list[-1].gas_phase.sum() == pytest.approx(9585.032869548137))
    assert(co2_data.data_list[-1].aqu_phase.sum() == pytest.approx(2834.956447728449))
