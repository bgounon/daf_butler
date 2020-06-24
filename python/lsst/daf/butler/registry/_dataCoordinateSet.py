# This file is part of daf_butler.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = (
    "DataCoordinateSet",
)

from typing import (
    AbstractSet,
    Any,
    Iterator,
    Type,
    TypeVar,
)

from ..core import (
    DataCoordinate,
    DimensionGraph,
)
from ._dataCoordinateIterable import DataCoordinateIterable


D = TypeVar("D", bound=DataCoordinate)


class DataCoordinateSet(DataCoordinateIterable[D]):

    __slots__ = ("_graph", "_dtype", "_native")

    def __init__(self, graph: DimensionGraph, native: AbstractSet[D], *,
                 dtype: Type[DataCoordinate] = DataCoordinate, check: bool = False):
        self._graph = graph
        self._dtype = dtype
        self._native = native
        if check:
            for dataId in self._native:
                if not isinstance(dataId, self._dtype):
                    raise TypeError(f"Bad DataCoordinate subclass instance '{type(dataId).__name__}'; "
                                    f"{self._dtype.__name__} required in this context.")
                if dataId.graph != self._graph:
                    raise ValueError(f"Bad dimensions {dataId.graph}; expected {self._graph}.")

    def __iter__(self) -> Iterator[D]:
        return iter(self._native)

    def __len__(self) -> int:
        return len(self._native)

    def __contains__(self, key: Any) -> bool:
        key = DataCoordinate.standardize(key, graph=self.graph)
        return key in self._native

    def __str__(self) -> str:
        return str(self._native)

    def __repr__(self) -> str:
        return f"DataCoordinateSet({self._graph!r}, {self._native!r}, dtype={self._dtype})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataCoordinateSet):
            return (
                self._graph == other._graph
                and self._dtype == other._dtype
                and self._native == other._native
            )
        return False

    def __le__(self, other: DataCoordinateSet[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native <= other._native

    def __ge__(self, other: DataCoordinateSet[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native >= other._native

    def __lt__(self, other: DataCoordinateSet[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native < other._native

    def __gt__(self, other: DataCoordinateSet[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native > other._native

    def issubset(self, other: DataCoordinateIterable[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native <= other.toSet()._native

    def issuperset(self, other: DataCoordinateIterable[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native >= other.toSet()._native

    def isdisjoint(self, other: DataCoordinateIterable[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native.isdisjoint(other.toSet()._native)

    def __and__(self, other: DataCoordinateSet[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self.graph, self._native & other._native, dtype=self.dtype)

    def __or__(self, other: DataCoordinateSet[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self.graph, self._native | other._native, dtype=self.dtype)

    def __xor__(self, other: DataCoordinateSet[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self.graph, self._native ^ other._native, dtype=self.dtype)

    def __sub__(self, other: DataCoordinateSet[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self.graph, self._native - other._native, dtype=self.dtype)

    def intersection(self, other: DataCoordinateIterable[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self.graph, self._native & other.toSet()._native, dtype=self.dtype)

    def union(self, other: DataCoordinateIterable[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self.graph, self._native | other.toSet()._native, dtype=self.dtype)

    def symmetric_difference(self, other: DataCoordinateIterable[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self.graph, self._native ^ other.toSet()._native, dtype=self.dtype)

    def difference(self, other: DataCoordinateIterable[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self.graph, self._native - other.toSet()._native, dtype=self.dtype)

    def toSet(self) -> DataCoordinateSet[D]:
        return self

    @property
    def graph(self) -> DimensionGraph:
        return self._graph

    @property
    def dtype(self) -> Type[DataCoordinate]:
        return self._dtype

    def subset(self, graph: DimensionGraph) -> DataCoordinateSet[D]:
        if not graph.issubset(self.graph):
            raise ValueError(f"{graph} is not a subset of {self.graph}")
        if graph == self.graph:
            return self
        # See comment in _ScalarDataCoordinateIterable on type: ignore there.
        return DataCoordinateSet(
            graph,
            {dataId.subset(graph) for dataId in self._native}  # type: ignore
        )
