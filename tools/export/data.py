# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PointPillars export APIs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import logging

import h5py
import numpy as np
"""
__init__: 构造函数，用于初始化 TensorFile 对象。接受以下参数：

filename: HDF5文件的路径。
mode: 打开文件的模式（与Python的标准文件打开模式类似）。
enforce_same_shape: 布尔值，如果为True，则强制所有写入的张量具有相同的形状。
_get_group_name: 内部方法，用于基于游标索引生成HDF5数据集的名称。

_write_data: 内部方法，用于将一个numpy数组或包含numpy数组的字典写入HDF5文件。

close: 关闭HDF5文件的方法。

next (Python 2) 和 __next__ (Python 3): 迭代方法，允许使用迭代器遍历文件中的所有元素。

_read_data: 内部方法，用于读取HDF5组中的数据，并将其转换回字典或numpy数组。

read: 读取当前游标位置的数据。如果到达文件末尾，则返回None。

readable: 返回实例是否可读。

seekable: 返回实例是否可寻址。

seek: 移动读写游标到指定位置。

tell: 返回当前游标的位置。

truncate: 不支持的方法，因为HDF5不支持标准的截断操作。

writable: 返回实例是否可写。

write: 将一个numpy数组或包含numpy数组的字典写入文件。
"""

"""Logger for data export APIs."""
logger = logging.getLogger(__name__)

"""这个 data.py 文件定义了一个名为 TensorFile 的Python类，此类的目的是提供一个API，用于将多个张量（通常是numpy数组）读写到文件中。它使用HDF5数据库格式存储数据，HDF5是一个用于存储和组织大量数据的文件格式。"""
class TensorFile(io.RawIOBase):
    """Class to read/write multiple tensors to a file.

    The underlying implementation using an HDF5 database
    to store data.

    Note: this class does not support multiple writers to
    the same file.

    Args:
        filename (str): path to file.
        mode (str): mode to open file in.
            r   Readonly, file must exist
            r+  Read/write, file must exist
            w   Create file, truncate if exists
            w-  Create file, fail if exists
            a   Read/write if exists, create otherwise (default)
        enforce_same_shape (bool): whether to enforce that all tensors be the same shape.
    """

    DEFAULT_ARRAY_KEY = "_tensorfile_array_key_"
    GROUP_NAME_PREFIX = "_tensorfile_array_key_"
    """
    __init__: 构造函数，用于初始化 TensorFile 对象。接受以下参数：

        filename: HDF5文件的路径。
        mode: 打开文件的模式（与Python的标准文件打开模式类似）。
        enforce_same_shape: 布尔值，如果为True，则强制所有写入的张量具有相同的形状。
    """
    def __init__(
        self, filename, mode="a", enforce_same_shape=True, *args, **kwargs
    ):  # pylint: disable=W1113
        """Init routine."""
        super(TensorFile, self).__init__(*args, **kwargs)

        logger.debug("Opening %s with mode=%s", filename, mode)

        self._enforce_same_shape = enforce_same_shape
        self._mode = mode

        # Open or create the HDF5 file.
        self._db = h5py.File(filename, mode)

        if "count" not in self._db.attrs:
            self._db.attrs["count"] = 0

        if "r" in mode:
            self._cursor = 0
        else:
            self._cursor = self._db.attrs["count"]

    def _get_group_name(cls, cursor):
        """Return the name of the H5 dataset to create, given a cursor index."""
        return "%s_%d" % (cls.GROUP_NAME_PREFIX, cursor)

    def _write_data(self, group, data):
        for key, value in data.items():
            if isinstance(value, dict):
                self._write_data(group.create_group(key), value)
            elif isinstance(value, np.ndarray):
                if self._enforce_same_shape:
                    if "shape" not in self._db.attrs:
                        self._db.attrs["shape"] = value.shape
                    else:
                        expected_shape = tuple(self._db.attrs["shape"].tolist())
                        if expected_shape != value.shape:
                            raise ValueError(
                                "Shape mismatch: %s v.s. %s"
                                % (str(expected_shape), str(value.shape))
                            )
                group.create_dataset(key, data=value, compression="gzip")
            else:
                raise ValueError(
                    "Only np.ndarray or dicts can be written into a TensorFile."
                )

    def close(self):
        """Close this file."""
        self._db.close()

    # For python2.
    def next(self):
        """Return next element."""
        return self.__next__()

    # For python3.
    def __next__(self):
        """Return next element."""
        if self._cursor < self._db.attrs["count"]:
            return self.read()
        raise StopIteration()

    def _read_data(self, group):
        if isinstance(group, h5py.Group):
            data = {key: self._read_data(value) for key, value in group.items()}
        else:
            data = group[()]
        return data

    def read(self):
        """Read from current cursor.

        Return array assigned to current cursor, or ``None`` to indicate
        the end of the file.
        """
        if not self.readable():
            raise IOError("Instance is not readable.")

        group_name = self._get_group_name(self._cursor)

        if group_name in self._db:
            self._cursor += 1
            group = self._db[group_name]
            data = self._read_data(group)
            if list(data.keys()) == [self.DEFAULT_ARRAY_KEY]:
                # The only key in this group is the default key.
                # Return the numpy array directly.
                return data[self.DEFAULT_ARRAY_KEY]
            return data
        return None

    def readable(self):
        """Return whether this instance is readable."""
        return self._mode in ["r", "r+", "a"]

    def seekable(self):
        """Return whether this instance is seekable."""
        return True

    def seek(self, n):
        """Move cursor."""
        self._cursor = min(n, self._db.attrs["count"])
        return self._cursor

    def tell(self):
        """Return current cursor index."""
        return self._cursor

    def truncate(self, n):
        """Truncation is not supported."""
        raise IOError("Truncate operation is not supported.")

    def writable(self):
        """Return whether this instance is writable."""
        return self._mode in ["r+", "w", "w-", "a"]

    def write(self, data):
        """Write a Numpy array or a dictionary of numpy arrays into file."""
        if not self.writable():
            raise IOError("Instance is not writable.")

        if isinstance(data, np.ndarray):
            data = {self.DEFAULT_ARRAY_KEY: data}

        group_name = self._get_group_name(self._cursor)

        # Delete existing instance of datasets at this cursor position.
        if group_name in self._db:
            del self._db[group_name]

        group = self._db.create_group(group_name)

        self._write_data(group, data)

        self._cursor += 1

        if self._cursor > self._db.attrs["count"]:
            self._db.attrs["count"] = self._cursor
