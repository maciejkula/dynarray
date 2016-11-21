# Dynarray

[![CircleCI](https://circleci.com/gh/maciejkula/dynarray.svg?style=svg)](https://circleci.com/gh/maciejkula/dynarray)


Dynamically growable Numpy arrays. They function exactly like normal numpy arrays, but support appending new elements.

# Installation

Simply install from PyPI: `pip install dynarray`


# Quickstart

Create an empty one-dimensional array and append elements to it:

```python
from dynarray import DynamicArray

array = DynamicArray()

for element in range(10):
    array.append(element)
```

Create a multidimensional array and append rows:

```python
from dynarray import DynamicArray

# The leading dimension is None to denote that this is
# the dynamic dimension
array = DynamicArray((None, 20, 10))

array.append(np.random.random((20, 10)))
array.extend(np.random.random((100, 20, 10)))

print(array.shape)  # (101, 20, 10)
```

Slice and perform arithmetic like with normal numpy arrays:

```python
from dynarray import DynamicArray

array = DynamicArray(np.ones((100, 10)), dtype=np.float16)

assert array[10:11].sum() == 10.0
print(array[10])
array[10] *= 0.0
assert array[10].sum() == 0.0
```

Shrink to fit to minimize memory usage when no further resizing is needed:

```python
from dynarray import DynamicArray

array = DynamicArray(np.ones((100, 10)), dtype=np.float16)
array.extend(np.ones((50, 10)))

assert array.capacity == 200
array.shrink_to_fit()
assert array.capacity == 150
```
