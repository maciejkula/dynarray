import itertools

from hypothesis import given
from hypothesis.strategies import integers, lists, sampled_from
from hypothesis.extra.numpy import arrays

import numpy as np

import pytest

from dynarray import DynamicArray


def arrays_strategy():

    shapes = lists(integers(min_value=1, max_value=4),
                   min_size=1, max_size=4)

    possible_dtypes = [x for x in itertools.chain(*np.sctypes.values())
                       if x not in (np.void, np.object_, np.object)]
    dtypes = sampled_from(possible_dtypes)

    return shapes.flatmap(
        lambda shape: dtypes.flatmap(
            lambda dtype: arrays(dtype, shape)))


@given(arrays_strategy())
def test_appending(source_array):

    dtype = source_array.dtype
    input_arg = source_array.shape[1:]

    array = DynamicArray(input_arg, dtype)

    for row in source_array:
        array.append(row)

    try:
        assert np.all(np.logical_or(np.isnan(source_array),
                                    array[:] == source_array))
    except TypeError:
        assert np.all(array[:] == source_array)
