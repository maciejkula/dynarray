import itertools

from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import integers, lists, sampled_from
from hypothesis.extra.numpy import arrays

import numpy as np

from dynarray import DynamicArray


def arrays_strategy():

    shapes = lists(integers(min_value=1, max_value=15),
                   min_size=1, max_size=4)

    possible_dtypes = [x for x in itertools.chain(*np.sctypes.values())
                       if x not in (np.void, np.object_, np.object)]
    dtypes = sampled_from(possible_dtypes)

    return shapes.flatmap(
        lambda shape: dtypes.flatmap(
            lambda dtype: arrays(dtype, shape)))


def assert_equal_or_nan(x, y):

    assert np.all(np.logical_or(x == y,
                                np.isnan(x)))


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(arrays_strategy())
def test_appending(source_array):

    dtype = source_array.dtype
    input_arg = source_array.shape[1:]

    array = DynamicArray(input_arg, dtype, allow_views_on_resize=True)

    _ = array[:]  # NOQA

    for row in source_array:
        array.append(row)

    try:
        assert np.all(np.logical_or(np.isnan(source_array),
                                    array[:] == source_array))
    except TypeError:
        assert np.all(array[:] == source_array)


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(arrays_strategy())
def test_appending_with_views_fails(source_array):

    dtype = source_array.dtype
    input_arg = source_array.shape[1:]

    array = DynamicArray(input_arg, dtype)
    array.shrink_to_fit()  # Force a reallocation on first append

    _ = array[:]  # NOQA

    try:
        for row in source_array:
            array.append(row)
        assert False, 'An exception should have been raised.'
    except ValueError as e:
        assert 'allow_views_on_resize' in e.message


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(arrays_strategy())
def test_appending_lists(source_array):

    dtype = source_array.dtype
    input_arg = source_array.shape[1:]

    array = DynamicArray(input_arg, dtype, allow_views_on_resize=True)

    _ = array[:]  # NOQA

    for row in source_array:
        row_list = row.tolist()

        if isinstance(row_list, basestring):
            # Numpy has problems parsing unicode
            return

        array.append(row_list)

    try:
        assert np.all(np.logical_or(np.isnan(source_array),
                                    array[:] == source_array))
    except TypeError:
        assert np.all(array[:] == source_array)


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(arrays_strategy())
def test_extending(source_array):

    dtype = source_array.dtype
    input_arg = source_array.shape[1:]

    array = DynamicArray(input_arg, dtype)

    for row in source_array:
        array.extend(source_array)

    comparison_array = np.concatenate([source_array] * source_array.shape[0])

    try:
        assert np.all(np.logical_or(np.isnan(comparison_array),
                                    array[:] == comparison_array))
    except TypeError:
        assert np.all(array[:] == comparison_array)


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(arrays_strategy())
def test_array_constructor(source_array):

    array = DynamicArray(source_array)

    try:
        assert np.all(np.logical_or(np.isnan(source_array),
                                    array[:] == source_array))
    except TypeError:
        assert np.all(array[:] == source_array)


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(arrays_strategy())
def test_extending_array_constructor(source_array):

    array = DynamicArray(source_array)

    for row in source_array:
        array.extend(source_array)

    comparison_array = np.concatenate([source_array] *
                                      (1 + source_array.shape[0]))

    try:
        assert np.all(np.logical_or(np.isnan(comparison_array),
                                    array[:] == comparison_array))
    except TypeError:
        assert np.all(array[:] == comparison_array)


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(arrays_strategy())
def test_attr_delegation(source_array):

    array = DynamicArray(source_array)

    try:
        source_array + source_array
    except TypeError:
        return

    try:
        assert_equal_or_nan(array + source_array,
                            source_array + source_array)
    except TypeError:
        assert (array + source_array ==
                source_array + source_array)

    try:
        assert_equal_or_nan(array - source_array,
                            source_array - source_array)
    except TypeError:
        assert (array - source_array ==
                source_array - source_array)

    try:
        assert_equal_or_nan(array * source_array,
                            source_array * source_array)
    except TypeError:
        assert (array * source_array ==
                source_array * source_array)

    try:
        assert_equal_or_nan(array / source_array,
                            source_array / source_array)
    except TypeError:
        assert (array / source_array ==
                source_array / source_array)

    try:
        assert_equal_or_nan(array ** 2,
                            source_array ** 2)
    except TypeError:
        assert (array ** 2 ==
                source_array ** 2)

    # In-place operators
    try:
        array = DynamicArray(source_array)
        array += source_array
        assert_equal_or_nan(array,
                            source_array + source_array)
    except TypeError:
        array += source_array
        assert (array ==
                source_array + source_array)

    try:
        array = DynamicArray(source_array)
        array -= source_array
        assert_equal_or_nan(array,
                            source_array - source_array)
    except TypeError:
        array -= source_array
        assert (array ==
                source_array - source_array)

    try:
        array = DynamicArray(source_array)
        array *= source_array
        assert_equal_or_nan(array,
                            source_array * source_array)
    except TypeError:
        array *= source_array
        assert (array ==
                source_array * source_array)

    try:
        array = DynamicArray(source_array)
        array /= source_array
        assert_equal_or_nan(array,
                            source_array / source_array)
    except TypeError:
        array /= source_array
        assert (array ==
                source_array / source_array)
