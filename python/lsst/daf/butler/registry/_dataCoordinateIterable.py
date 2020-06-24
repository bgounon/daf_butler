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
    "DataCoordinateIterable",
)

from abc import abstractmethod
from typing import (
    AbstractSet,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    TYPE_CHECKING,
    Type,
    TypeVar,
)

import sqlalchemy

from ..core import (
    DataCoordinate,
    DimensionGraph,
    MinimalDataCoordinate,
)

if TYPE_CHECKING:
    from .simpleQuery import SimpleQuery
    from ._dataCoordinateSet import DataCoordinateSet


D = TypeVar("D", bound=DataCoordinate)


class DataCoordinateIterable(Iterable[D]):

    __slots__ = ()

    @staticmethod
    def fromScalar(dataId: D) -> _ScalarDataCoordinateIterable[D]:
        return _ScalarDataCoordinateIterable(dataId)

    @staticmethod
    def fromSet(dataIds: AbstractSet[D], graph: DimensionGraph, *,
                dtype: Type[DataCoordinate] = DataCoordinate) -> DataCoordinateSet[D]:
        from ._dataCoordinateSet import DataCoordinateSet
        return DataCoordinateSet(graph, dataIds, dtype=dtype)

    @property
    @abstractmethod
    def graph(self) -> DimensionGraph:
        """The dimensions identified by thes data IDs (`DimensionGraph`).
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def dtype(self) -> Type[DataCoordinate]:
        # TODO: docs
        raise NotImplementedError()

    def minimal(self) -> DataCoordinateIterable[MinimalDataCoordinate]:
        # TODO: docs
        if issubclass(self.dtype, MinimalDataCoordinate):
            return self  # type: ignore
        else:
            return _MinimalViewDataCoordinateIterable(self)

    def toSet(self) -> DataCoordinateSet[D]:
        # TODO: docs
        from ._dataCoordinateSet import DataCoordinateSet
        return DataCoordinateSet(self.graph, set(self), dtype=self.dtype)

    def constrain(self, query: SimpleQuery, columns: Callable[[str], sqlalchemy.sql.ColumnElement]) -> None:
        # TODO: docs
        toOrTogether: List[sqlalchemy.sql.ColumnElement] = []
        for dataId in self:
            toOrTogether.append(
                sqlalchemy.sql.and_(*[
                    columns(dimension.name) == dataId[dimension.name]
                    for dimension in self.graph.required
                ])
            )
        query.where.append(sqlalchemy.sql.or_(*toOrTogether))

    @abstractmethod
    def subset(self, graph: DimensionGraph) -> DataCoordinateIterable[D]:
        raise NotImplementedError()


class _ScalarDataCoordinateIterable(DataCoordinateIterable[D]):

    __slots__ = ("_dataId",)

    def __init__(self, dataId: D):
        self._dataId = dataId

    def __iter__(self) -> Iterator[D]:
        yield self._dataId

    def __len__(self) -> int:
        return 1

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, DataCoordinate):
            return key == self._dataId
        else:
            return False

    @property
    def graph(self) -> DimensionGraph:
        return self._dataId.graph

    @property
    def dtype(self) -> Type[DataCoordinate]:
        return type(self._dataId)

    def subset(self, graph: DimensionGraph) -> _ScalarDataCoordinateIterable[D]:
        # No good way to tell MyPy the return type is covariant here (i.e. that
        # we know D.subset(graph) -> D when D is one of Minimal-, Complete- or
        # ExtendedDataCoordinate); we can't use generics in
        # DataCoordinate.subset because that would demand that all
        # implementations return their own type, not just their most
        # appropriate ABC.
        return _ScalarDataCoordinateIterable(self._dataId.subset(graph))  # type: ignore


class _MinimalViewDataCoordinateIterable(DataCoordinateIterable[MinimalDataCoordinate], Generic[D]):

    __slots__ = ("_target",)

    def __init__(self, target: DataCoordinateIterable[D]):
        self._target = target

    def __iter__(self) -> Iterator[MinimalDataCoordinate]:
        for dataId in self._target:
            yield dataId.minimal()

    @property
    def graph(self) -> DimensionGraph:
        return self._target.graph

    @property
    def dtype(self) -> Type[DataCoordinate]:
        return MinimalDataCoordinate

    def minimal(self) -> _MinimalViewDataCoordinateIterable:
        return self

    def subset(self, graph: DimensionGraph) -> DataCoordinateIterable[MinimalDataCoordinate]:
        return self._target.subset(graph).minimal()
