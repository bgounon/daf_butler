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
    "DataCoordinate",
    "CompleteDataCoordinate",
    "EmptyDataCoordinate",
    "ExpandedDataCoordinate",
    "MinimalDataCoordinate",
    "DataId",
    "DataIdKey",
    "DataIdValue",
)

from abc import ABC, abstractmethod
import numbers
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import astropy.time

from lsst.sphgeom import Region
from ..named import NamedKeyMapping, NamedValueSet
from ..timespan import Timespan
from .elements import Dimension, DimensionElement
from .graph import DimensionGraph

if TYPE_CHECKING:  # Imports needed only for type annotations; may be circular.
    from .universe import DimensionUniverse
    from .records import DimensionRecord


DataIdKey = Union[str, Dimension]
DataIdValue = Union[str, int, None]


class DataCoordinate(ABC):

    __slots__ = ()

    @staticmethod
    def standardize(mapping: Optional[Union[Mapping[str, DataIdValue], DataCoordinate]] = None, *,
                    graph: Optional[DimensionGraph] = None,
                    universe: Optional[DimensionUniverse] = None,
                    **kwargs: Any) -> DataCoordinate:
        """Adapt an arbitrary mapping and/or additional arguments into a true
        `DataCoordinate`, or augment an existing one.

        Parameters
        ----------
        mapping : `~collections.abc.Mapping`, optional
            An informal data ID that maps dimension names to their primary key
            values (may also be a true `DataCoordinate`).
        graph : `DimensionGraph`
            The dimensions to be identified by the new `DataCoordinate`.
            If not provided, will be inferred from the keys of ``mapping``,
            and ``universe`` must be provided unless ``mapping`` is already a
            `DataCoordinate`.
        universe : `DimensionUniverse`
            All known dimensions and their relationships; used to expand
            and validate dependencies when ``graph`` is not provided.
        **kwargs
            Additional keyword arguments are treated like additional key-value
            pairs in ``mapping``.

        Returns
        -------
        coordinate : `DataCoordinate`
            A validated `DataCoordinate` instance.  Will be a
            `CompleteDataCoordinate` if all implied dimensions are identified,
            an `ExpandedDataCoordinate` if ``mapping`` is already an
            `ExpandedDataCoordinate` and ``kwargs`` is empty, and a
            `MinimalDataCoordinate` otherwise.

        Raises
        ------
        TypeError
            Raised if the set of optional arguments provided is not supported.
        KeyError
            Raised if a key-value pair for a required dimension is missing.

        Notes
        -----
        Because `MinimalDataCoordinate` stores only values for required
        dimensions, key-value pairs for other implied dimensions will be
        ignored and excluded from the result unless _all_ implied dimensions
        are identified (and hence a `CompleteDataCoordinate` can be returned).
        This means that a `DataCoordinate` may contain *fewer* key-value pairs
        than the informal data ID dictionary it was constructed from.
        """
        d: Dict[str, DataIdValue] = {}
        if isinstance(mapping, DataCoordinate):
            if graph is None:
                if not kwargs:
                    # Already standardized to exactly what we want.
                    return mapping
            elif (isinstance(mapping, CompleteDataCoordinate)
                    and kwargs.keys().isdisjoint(graph.dimensions.names)):
                # User provided kwargs, but told us not to use them by
                # passing in a disjoint graph.
                return mapping.subset(graph)
            assert universe is None or universe == mapping.universe
            universe = mapping.universe
            d.update((name, mapping[name]) for name in mapping.graph.required.names)
            if isinstance(mapping, CompleteDataCoordinate):
                d.update((name, mapping[name]) for name in mapping.graph.implied.names)
        elif mapping is not None:
            d.update(mapping)
        d.update(kwargs)
        if graph is None:
            if universe is None:
                raise TypeError("universe must be provided if graph is not.")
            graph = DimensionGraph(universe, names=d.keys())
        if not graph.dimensions:
            return DataCoordinate.makeEmpty(graph.universe)
        cls: Callable[[DimensionGraph, Tuple[DataIdValue, ...]], DataCoordinate]
        if d.keys() >= graph.dimensions.names:
            cls = CompleteDataCoordinate.fromValues
            values = tuple(d[name] for name in graph.dimensions.names)
        else:
            cls = MinimalDataCoordinate.fromValues
            try:
                values = tuple(d[name] for name in graph.required.names)
            except KeyError as err:
                raise KeyError(f"No value in data ID ({mapping}) for required dimension {err}.") from err
        # Some backends cannot handle numpy.int64 type which is a subclass of
        # numbers.Integral; convert that to int.
        values = tuple(int(val) if isinstance(val, numbers.Integral)  # type: ignore
                       else val for val in values)
        return cls(graph, values)

    @staticmethod
    def makeEmpty(universe: DimensionUniverse) -> EmptyDataCoordinate:
        return EmptyDataCoordinate(universe)

    @abstractmethod
    def __getitem__(self, key: DataIdKey) -> DataIdValue:
        raise NotImplementedError()

    def get(self, key: DataIdKey, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    @property
    @abstractmethod
    def graph(self) -> DimensionGraph:
        """The dimensions identified by this data ID (`DimensionGraph`).

        Note that values are only required to be present for dimensions in
        ``self.graph.required``; all others may be retrieved (from a
        `Registry`) given these.
        """
        raise NotImplementedError()

    def minimal(self) -> MinimalDataCoordinate:
        # TODO: docs
        return MinimalDataCoordinate.fromValues(
            self.graph,
            tuple(self[name] for name in self.graph.required.names)
        )

    @abstractmethod
    def subset(self, graph: DimensionGraph) -> DataCoordinate:
        """Return a new `DataCoordinate` whose graph is a subset of
        ``self.graph``.

        Subclasses may override this method to return a subclass instance or
        operate more efficiently.

        Parameters
        ----------
        graph : `DimensionGraph`
            The dimensions identified by the returned `DataCoordinate`.

        Returns
        -------
        coordinate : `DataCoordinate`
            A `DataCoordinate` instance that identifies only the given
            dimensions.

        Raises
        ------
        KeyError
            Raised if ``graph`` is not a subset of ``self.graph``, and hence
            one or more dimensions has no associated primary key value.
        NotImplementedError
            Raised if ``graph`` is a subset of ``self.graph``, but its
            required dimensions are only implied dimensions in ``self.graph``,
            and ``self`` is a `MinimalDataCoordinate` (and hence those
            dimension values are unknown).
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        # We can't make repr yield something that could be exec'd here without
        # printing out the whole DimensionUniverse the graph is derived from.
        # So we print something that mostly looks like a dict, but doesn't
        # quote its keys: that's both more compact and something that can't
        # be mistaken for an actual dict or something that could be exec'd.
        return "{{{}}}".format(
            ', '.join(f"{d}: {self.get(d, '?')!r}" for d in self.graph.dimensions.names)
        )

    def __hash__(self) -> int:
        return hash((self.graph,) + tuple(self[d.name] for d in self.graph.required))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataCoordinate):
            other = DataCoordinate.standardize(other)
        return self.graph == other.graph and all(self[d.name] == other[d.name] for d in self.graph.required)

    @property
    def universe(self) -> DimensionUniverse:
        """The universe that defines all known dimensions compatible with
        this coordinate (`DimensionUniverse`).
        """
        return self.graph.universe


class MinimalDataCoordinate(DataCoordinate, NamedKeyMapping[Dimension, DataIdValue]):
    __slots__ = ()

    @staticmethod
    def fromValues(graph: DimensionGraph, values: Tuple[DataIdValue, ...]) -> MinimalDataCoordinate:
        return _MinimalTupleDataCoordinate(graph, values)

    @staticmethod
    def fromMapping(graph: DimensionGraph, mapping: Mapping[str, DataIdValue]) -> MinimalDataCoordinate:
        return MinimalDataCoordinate.fromValues(
            graph,
            tuple(mapping[name] for name in graph.required.names)
        )

    @abstractmethod
    def keys(self) -> NamedValueSet[Dimension]:
        raise NotImplementedError()

    def minimal(self) -> MinimalDataCoordinate:
        # Docstring inherited from DataCoordinate.
        return self

    def subset(self, graph: DimensionGraph) -> MinimalDataCoordinate:
        if graph == self.graph:
            return self
        if not (graph <= self.graph):
            raise KeyError(f"{graph} is not a subset of {self.graph}.")
        if not (graph.required <= self.graph.required):
            raise NotImplementedError(
                f"No value for implied dimension(s) {graph.required - self.graph.required} in "
                f"MinimalDataCoordinate {self}."
            )
        return _MinimalTupleDataCoordinate(
            graph,
            tuple(self[name] for name in graph.dimensions.names)
        )


class CompleteDataCoordinate(DataCoordinate, NamedKeyMapping[Dimension, DataIdValue]):
    __slots__ = ()

    @staticmethod
    def fromValues(graph: DimensionGraph, values: Tuple[DataIdValue, ...]) -> CompleteDataCoordinate:
        return _CompleteTupleDataCoordinate(graph, values)

    @staticmethod
    def fromMapping(graph: DimensionGraph, mapping: Mapping[str, DataIdValue]) -> CompleteDataCoordinate:
        return CompleteDataCoordinate.fromValues(
            graph,
            tuple(mapping[name] for name in graph.dimensions.names)
        )

    def expanded(self, records: Dict[str, Optional[DimensionRecord]]) -> ExpandedDataCoordinate:
        return _ExpandedTupleDataCoordinate(self.graph, tuple(self.values()), records=records)

    @abstractmethod
    def keys(self) -> NamedValueSet[Dimension]:
        raise NotImplementedError()

    def subset(self, graph: DimensionGraph) -> CompleteDataCoordinate:
        if graph == self.graph:
            return self
        return _CompleteTupleDataCoordinate(
            graph,
            tuple(self[name] for name in graph.dimensions.names)
        )


def _intersectRegions(*args: Region) -> Optional[Region]:
    """Return the intersection of several regions.

    For internal use by `ExpandedDataCoordinate` only.

    If no regions are provided, returns `None`.

    This is currently a placeholder; it actually returns `NotImplemented`
    (it does *not* raise an exception) when multiple regions are given, which
    propagates to `ExpandedDataCoordinate`.  This reflects the fact that we
    don't want to fail to construct an `ExpandedDataCoordinate` entirely when
    we can't compute its region, and at present we don't have a high-level use
    case for the regions of these particular data IDs.
    """
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        return NotImplemented


class ExpandedDataCoordinate(CompleteDataCoordinate):
    __slots__ = ()

    @abstractmethod
    def record(self, key: Union[DimensionElement, str]) -> Optional[DimensionRecord]:
        raise NotImplementedError()

    @property
    def region(self) -> Optional[Region]:
        # TODO: docs
        regions = []
        for element in self.graph.spatial:
            record = self.record(element.name)
            # DimensionRecord subclasses for spatial elements always have a
            # .region, but they're dynamic so this can't be type-checked.
            if record is None or record.region is None:  # type: ignore
                return None
            else:
                regions.append(record.region)  # type:ignore
        return _intersectRegions(*regions)

    @property
    def timespan(self) -> Optional[Timespan[astropy.time.Time]]:
        # TODO: docs
        timespans = []
        for element in self.graph.temporal:
            record = self.record(element.name)
            # DimensionRecord subclasses for temporal elements always have
            # .timespan, but they're dynamic so this can't be type-checked.
            if record is None or record.timespan is None:  # type:ignore
                return None
            else:
                timespans.append(record.timespan)  # type:ignore
        return Timespan.intersection(*timespans)

    def pack(self, name: str, *, returnMaxBits: bool = False) -> Union[Tuple[int, int], int]:
        """Pack this data ID into an integer.

        Parameters
        ----------
        name : `str`
            Name of the `DimensionPacker` algorithm (as defined in the
            dimension configuration).
        returnMaxBits : `bool`, optional
            If `True` (`False` is default), return the maximum number of
            nonzero bits in the returned integer across all data IDs.

        Returns
        -------
        packed : `int`
            Integer ID.  This ID is unique only across data IDs that have
            the same values for the packer's "fixed" dimensions.
        maxBits : `int`, optional
            Maximum number of nonzero bits in ``packed``.  Not returned unless
            ``returnMaxBits`` is `True`.
        """
        return self.universe.makePacker(name, self).pack(self, returnMaxBits=returnMaxBits)

    def subset(self, graph: DimensionGraph) -> ExpandedDataCoordinate:
        if graph == self.graph:
            return self
        return _ExpandedTupleDataCoordinate(
            graph,
            tuple(self[d] for d in graph.dimensions),
            records={e.name: self.record(e.name) for e in graph.elements},
        )


DataId = Union[DataCoordinate, Mapping[str, DataIdValue]]
"""A type-annotation alias for signatures that accept both informal data ID
dictionaries and validated `DataCoordinate` instances.
"""


class _DataCoordinateTupleMixin(NamedKeyMapping[Dimension, DataIdValue]):

    __slots__ = ("_graph", "_indices", "_values")

    def __init__(self, graph: DimensionGraph,
                 indices: Dict[str, int],
                 values: Tuple[DataIdValue, ...]):
        assert len(indices) == len(values)
        self._graph = graph
        self._indices = indices
        self._values = values

    @abstractmethod
    def keys(self) -> NamedValueSet[Dimension]:
        raise NotImplementedError()

    def values(self) -> Tuple[DataIdValue, ...]:  # type: ignore
        return self._values

    @property
    def names(self) -> AbstractSet[str]:
        return self.keys().names

    @property
    def graph(self) -> DimensionGraph:
        return self._graph

    def __getitem__(self, key: DataIdKey) -> DataIdValue:
        if isinstance(key, Dimension):
            key = key.name
        return self._values[self._indices[key]]

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[Dimension]:
        return iter(self.keys())


class _MinimalTupleDataCoordinate(_DataCoordinateTupleMixin, MinimalDataCoordinate):

    __slots__ = ()

    def __init__(self, graph: DimensionGraph, values: Tuple[DataIdValue, ...]):
        super().__init__(graph, graph._requiredIndices, values)

    def keys(self) -> NamedValueSet[Dimension]:
        return self.graph.required


class _CompleteTupleDataCoordinate(_DataCoordinateTupleMixin, CompleteDataCoordinate):

    __slots__ = ()

    def __init__(self, graph: DimensionGraph, values: Tuple[DataIdValue, ...]):
        super().__init__(graph, graph._dimensionIndices, values)

    def keys(self) -> NamedValueSet[Dimension]:
        return self.graph.dimensions


class _ExpandedTupleDataCoordinate(_CompleteTupleDataCoordinate, ExpandedDataCoordinate):

    __slots__ = ("_records",)

    def __init__(self, graph: DimensionGraph, values: Tuple[DataIdValue, ...], *,
                 records: Mapping[str, Optional[DimensionRecord]]):
        super().__init__(graph, values)
        self._records = records

    def record(self, key: Union[DimensionElement, str]) -> Optional[DimensionRecord]:
        # Docstring inherited from ExpandedDataCoordinate
        if isinstance(key, DimensionElement):
            return self._records[key.name]
        else:
            return self._records[key]


class EmptyDataCoordinate(MinimalDataCoordinate, ExpandedDataCoordinate):

    __slots__ = ("_universe",)

    def __init__(self, universe: DimensionUniverse):
        self._universe = universe

    @staticmethod
    def fromValues(graph: DimensionGraph, values: Tuple[DataIdValue, ...]) -> EmptyDataCoordinate:
        assert not graph and not values
        return EmptyDataCoordinate(graph.universe)

    @staticmethod
    def fromMapping(graph: DimensionGraph, mapping: Mapping[str, DataIdValue]) -> EmptyDataCoordinate:
        assert not graph
        return EmptyDataCoordinate(graph.universe)

    @property
    def graph(self) -> DimensionGraph:
        return self._universe.empty

    @property
    def universe(self) -> DimensionUniverse:
        return self._universe

    @property
    def names(self) -> AbstractSet[str]:
        return frozenset()

    def __getitem__(self, key: DataIdKey) -> DataIdValue:
        raise KeyError(f"Empty data ID indexed with {key}.")

    def keys(self) -> NamedValueSet[Dimension]:
        return NamedValueSet()

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[Dimension]:
        return iter(())

    def record(self, key: Union[DimensionElement, str]) -> Optional[DimensionRecord]:
        return None

    def subset(self, graph: DimensionGraph) -> EmptyDataCoordinate:
        if graph:
            raise KeyError(f"Cannot subset EmptyDataCoordinate with non-empty graph {graph}.")
        return self
