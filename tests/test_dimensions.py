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

import unittest
import copy
from dataclasses import dataclass
import os
import pickle
from random import Random
import itertools
from typing import Iterator, Optional

from lsst.daf.butler import (
    DatabaseTimespanRepresentation,
    DataCoordinate,
    DataCoordinateSequence,
    DataCoordinateSet,
    Dimension,
    DimensionGraph,
    DimensionUniverse,
    NamedKeyDict,
    NamedValueSet,
    Registry,
    YamlRepoImportBackend,
)
from lsst.daf.butler.registry import RegistryConfig

DIMENSION_DATA_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                    "data", "registry", "hsc-rc2-subset.yaml"))


def loadDimensionData() -> DataCoordinateSequence:
    """Load dimension data from an export file included in the code repository.

    Returns
    -------
    dataIds : `DataCoordinateSet`
        A set containing all data IDs in the export file.
    """
    # Create an in-memory SQLite database and Registry just to import the YAML
    # data and retreive it as a set of DataCoordinate objects.
    config = RegistryConfig()
    config["db"] = "sqlite://"
    registry = Registry.fromConfig(config, create=True)
    with open(DIMENSION_DATA_FILE, 'r') as stream:
        backend = YamlRepoImportBackend(stream, registry)
    backend.register()
    backend.load(datastore=None)
    dimensions = DimensionGraph(registry.dimensions, names=["visit", "detector", "tract", "patch"])
    return registry.queryDataIds(dimensions).expanded().toSequence()


class DimensionTestCase(unittest.TestCase):
    """Tests for dimensions.

    All tests here rely on the content of ``config/dimensions.yaml``, either
    to test that the definitions there are read in properly or just as generic
    data for testing various operations.
    """

    def setUp(self):
        self.universe = DimensionUniverse()

    def checkGraphInvariants(self, graph):
        elements = list(graph.elements)
        for n, element in enumerate(elements):
            # Ordered comparisons on graphs behave like sets.
            self.assertLessEqual(element.graph, graph)
            # Ordered comparisons on elements correspond to the ordering within
            # a DimensionUniverse (topological, with deterministic
            # tiebreakers).
            for other in elements[:n]:
                self.assertLess(other, element)
                self.assertLessEqual(other, element)
            for other in elements[n + 1:]:
                self.assertGreater(other, element)
                self.assertGreaterEqual(other, element)
            if isinstance(element, Dimension):
                self.assertEqual(element.graph.required, element.required)
        self.assertEqual(DimensionGraph(self.universe, graph.required), graph)
        self.assertCountEqual(graph.required,
                              [dimension for dimension in graph.dimensions
                               if not any(dimension in other.graph.implied for other in graph.elements)])
        self.assertCountEqual(graph.implied, graph.dimensions - graph.required)
        self.assertCountEqual(graph.dimensions,
                              [element for element in graph.elements
                               if isinstance(element, Dimension)])
        self.assertCountEqual(graph.dimensions, itertools.chain(graph.required, graph.implied))
        # Check primary key traversal order: each element should follow any it
        # requires, and element that is implied by any other in the graph
        # follow at least one of those.
        seen = NamedValueSet()
        for element in graph.primaryKeyTraversalOrder:
            with self.subTest(required=graph.required, implied=graph.implied, element=element):
                seen.add(element)
                self.assertLessEqual(element.graph.required, seen)
                if element in graph.implied:
                    self.assertTrue(any(element in s.implied for s in seen))
        self.assertCountEqual(seen, graph.elements)
        # Test encoding and decoding of DimensionGraphs to bytes.
        encoded = graph.encode()
        self.assertEqual(len(encoded), self.universe.getEncodeLength())
        self.assertEqual(DimensionGraph.decode(encoded, universe=self.universe), graph)

    def testConfigRead(self):
        self.assertEqual(self.universe.dimensions.names,
                         {"instrument", "visit", "visit_system", "exposure", "detector",
                          "physical_filter", "abstract_filter", "subfilter", "calibration_label",
                          "skymap", "tract", "patch", "htm7", "htm9"})

    def testGraphs(self):
        self.checkGraphInvariants(self.universe.empty)
        self.checkGraphInvariants(self.universe)
        for element in self.universe.elements:
            self.checkGraphInvariants(element.graph)

    def testInstrumentDimensions(self):
        graph = DimensionGraph(self.universe, names=("exposure", "detector", "visit", "calibration_label"))
        self.assertCountEqual(graph.dimensions.names,
                              ("instrument", "exposure", "detector", "calibration_label",
                               "visit", "physical_filter", "abstract_filter", "visit_system"))
        self.assertCountEqual(graph.required.names, ("instrument", "exposure", "detector",
                                                     "calibration_label", "visit"))
        self.assertCountEqual(graph.implied.names, ("physical_filter", "abstract_filter", "visit_system"))
        self.assertCountEqual(graph.elements.names - graph.dimensions.names,
                              ("visit_detector_region", "visit_definition"))

    def testCalibrationDimensions(self):
        graph = DimensionGraph(self.universe, names=("calibration_label", "physical_filter", "detector"))
        self.assertCountEqual(graph.dimensions.names,
                              ("instrument", "detector", "calibration_label",
                               "physical_filter", "abstract_filter"))
        self.assertCountEqual(graph.required.names, ("instrument", "detector", "calibration_label",
                                                     "physical_filter"))
        self.assertCountEqual(graph.implied.names, ("abstract_filter",))
        self.assertCountEqual(graph.elements.names, graph.dimensions.names)

    def testObservationDimensions(self):
        graph = DimensionGraph(self.universe, names=("exposure", "detector", "visit"))
        self.assertCountEqual(graph.dimensions.names, ("instrument", "detector", "visit", "exposure",
                                                       "physical_filter", "abstract_filter", "visit_system"))
        self.assertCountEqual(graph.required.names, ("instrument", "detector", "exposure", "visit"))
        self.assertCountEqual(graph.implied.names, ("physical_filter", "abstract_filter", "visit_system"))
        self.assertCountEqual(graph.elements.names - graph.dimensions.names,
                              ("visit_detector_region", "visit_definition"))
        self.assertCountEqual(graph.spatial.names, ("visit_detector_region",))
        self.assertCountEqual(graph.temporal.names, ("exposure",))

    def testSkyMapDimensions(self):
        graph = DimensionGraph(self.universe, names=("patch",))
        self.assertCountEqual(graph.dimensions.names, ("skymap", "tract", "patch"))
        self.assertCountEqual(graph.required.names, ("skymap", "tract", "patch"))
        self.assertCountEqual(graph.implied.names, ())
        self.assertCountEqual(graph.elements.names, graph.dimensions.names)
        self.assertCountEqual(graph.spatial.names, ("patch",))

    def testSubsetCalculation(self):
        """Test that independent spatial and temporal options are computed
        correctly.
        """
        graph = DimensionGraph(self.universe, names=("visit", "detector", "tract", "patch", "htm7",
                                                     "exposure", "calibration_label"))
        self.assertCountEqual(graph.spatial.names,
                              ("visit_detector_region", "patch", "htm7"))
        self.assertCountEqual(graph.temporal.names,
                              ("exposure", "calibration_label"))

    def testSchemaGeneration(self):
        tableSpecs = NamedKeyDict({})
        for element in self.universe.elements:
            if element.hasTable and element.viewOf is None:
                tableSpecs[element] = element.RecordClass.fields.makeTableSpec(
                    tsRepr=DatabaseTimespanRepresentation.Compound
                )
        for element, tableSpec in tableSpecs.items():
            for dep in element.required:
                with self.subTest(element=element.name, dep=dep.name):
                    if dep != element:
                        self.assertIn(dep.name, tableSpec.fields)
                        self.assertEqual(tableSpec.fields[dep.name].dtype, dep.primaryKey.dtype)
                        self.assertEqual(tableSpec.fields[dep.name].length, dep.primaryKey.length)
                        self.assertEqual(tableSpec.fields[dep.name].nbytes, dep.primaryKey.nbytes)
                        self.assertFalse(tableSpec.fields[dep.name].nullable)
                        self.assertTrue(tableSpec.fields[dep.name].primaryKey)
                    else:
                        self.assertIn(element.primaryKey.name, tableSpec.fields)
                        self.assertEqual(tableSpec.fields[element.primaryKey.name].dtype,
                                         dep.primaryKey.dtype)
                        self.assertEqual(tableSpec.fields[element.primaryKey.name].length,
                                         dep.primaryKey.length)
                        self.assertEqual(tableSpec.fields[element.primaryKey.name].nbytes,
                                         dep.primaryKey.nbytes)
                        self.assertFalse(tableSpec.fields[element.primaryKey.name].nullable)
                        self.assertTrue(tableSpec.fields[element.primaryKey.name].primaryKey)
            for dep in element.implied:
                with self.subTest(element=element.name, dep=dep.name):
                    self.assertIn(dep.name, tableSpec.fields)
                    self.assertEqual(tableSpec.fields[dep.name].dtype, dep.primaryKey.dtype)
                    self.assertFalse(tableSpec.fields[dep.name].primaryKey)
            for foreignKey in tableSpec.foreignKeys:
                self.assertIn(foreignKey.table, tableSpecs)
                self.assertIn(foreignKey.table, element.graph.dimensions.names)
                self.assertEqual(len(foreignKey.source), len(foreignKey.target))
                for source, target in zip(foreignKey.source, foreignKey.target):
                    self.assertIn(source, tableSpec.fields.names)
                    self.assertIn(target, tableSpecs[foreignKey.table].fields.names)
                    self.assertEqual(tableSpec.fields[source].dtype,
                                     tableSpecs[foreignKey.table].fields[target].dtype)
                    self.assertEqual(tableSpec.fields[source].length,
                                     tableSpecs[foreignKey.table].fields[target].length)
                    self.assertEqual(tableSpec.fields[source].nbytes,
                                     tableSpecs[foreignKey.table].fields[target].nbytes)

    def testPickling(self):
        # Pickling and copying should always yield the exact same object within
        # a single process (cross-process is impossible to test here).
        universe1 = DimensionUniverse()
        universe2 = pickle.loads(pickle.dumps(universe1))
        universe3 = copy.copy(universe1)
        universe4 = copy.deepcopy(universe1)
        self.assertIs(universe1, universe2)
        self.assertIs(universe1, universe3)
        self.assertIs(universe1, universe4)
        for element1 in universe1.elements:
            element2 = pickle.loads(pickle.dumps(element1))
            self.assertIs(element1, element2)
            graph1 = element1.graph
            graph2 = pickle.loads(pickle.dumps(graph1))
            self.assertIs(graph1, graph2)


@dataclass
class SplitByStateFlags:
    """A struct that separates data IDs with different states but the same
    values.
    """

    minimal: Optional[DataCoordinateSequence] = None
    """Data IDs that only contain values for required dimensions.

    `DataCoordinateSequence.hasFull()` will return `True` for this if and only
    if ``minimal.graph.implied`` has no elements.
    `DataCoordinate.hasRecords()` will always return `False`.
    """

    complete: Optional[DataCoordinateSequence] = None
    """Data IDs that contain values for all dimensions.

    `DataCoordinateSequence.hasFull()` will always `True` and
    `DataCoordinate.hasRecords()` will always return `True` for this attribute.
    """

    expanded: Optional[DataCoordinateSequence] = None
    """Data IDs that contain values for all dimensions as well as records.

    `DataCoordinateSequence.hasFull()` and `DataCoordinate.hasRecords()` will
    always return `True` for this attribute.
    """

    def chain(self, n: Optional[int] = None) -> Iterator:
        """Iterate over the data IDs of different types.

        Parameters
        ----------
        n : `int`, optional
            If provided (`None` is default), iterate over only the ``nth``
            data ID in each attribute.

        Yields
        ------
        dataId : `DataCoordinate`
            A data ID from one of the attributes in this struct.
        """
        if n is None:
            s = slice(None, None)
        else:
            s = slice(n, n + 1)
        if self.minimal is not None:
            yield from self.minimal[s]
        if self.complete is not None:
            yield from self.complete[s]
        if self.expanded is not None:
            yield from self.expanded[s]


class DataCoordinateTestCase(unittest.TestCase):

    RANDOM_SEED = 10

    @classmethod
    def setUpClass(cls):
        cls.allDataIds = loadDimensionData()

    def setUp(self):
        self.rng = Random(self.RANDOM_SEED)

    def randomDataIds(self, n: int, dataIds: Optional[DataCoordinateSequence] = None):
        """Select random data IDs from those loaded from test data.

        Parameters
        ----------
        n : `int`
             Number of data IDs to select.
        dataIds : `DataCoordinateSequence`, optional
            Data IDs to select from.  Defaults to ``self.allDataIds``.

        Returns
        -------
        selected : `DataCoordinateSequence`
            ``n`` Data IDs randomly selected from ``dataIds`` with replacement.
        """
        if dataIds is None:
            dataIds = self.allDataIds
        return DataCoordinateSequence(self.rng.sample(dataIds, n),
                                      graph=dataIds.graph,
                                      hasFull=dataIds.hasFull(),
                                      hasRecords=dataIds.hasRecords(),
                                      check=False)

    def randomDimensionSubset(self, n: int = 3, graph: Optional[DimensionGraph] = None) -> DimensionGraph:
        """Generate a random `DimensionGraph` that has a subset of the
        dimensions in a given one.

        Parameters
        ----------
        n : `int`
             Number of dimensions to select, before automatic expansion by
             `DimensionGraph`.
        dataIds : `DimensionGraph`, optional
            Dimensions to select ffrom.  Defaults to ``self.allDataIds.graph``.

        Returns
        -------
        selected : `DimensionGraph`
            ``n`` or more dimensions randomly selected from ``graph`` with
            replacement.
        """
        if graph is None:
            graph = self.allDataIds.graph
        return DimensionGraph(
            graph.universe,
            names=self.rng.sample(list(graph.dimensions.names), max(n, len(graph.dimensions)))
        )

    def splitByStateFlags(self, dataIds: Optional[DataCoordinateSequence] = None, *,
                          expanded: bool = True,
                          complete: bool = True,
                          minimal: bool = True) -> SplitByStateFlags:
        """Given a sequence of data IDs, generate new equivalent sequences
        containing less information.

        Parameters
        ----------
        dataIds : `DataCoordinateSequence`, optional.
            Data IDs to start from.  Defaults to ``self.allDataIds``.
            ``dataIds.hasRecords()`` and ``dataIds.hasFull()`` must both return
            `True`.
        expanded : `bool`, optional
            If `True` (default) include the original data IDs that contain all
            information in the result.
        complete : `bool`, optional
            If `True` (default) include data IDs for which ``hasFull()``
            returns `True` but ``hasRecords()`` does not.
        minimal : `bool`, optional
            If `True` (default) include data IDS that only contain values for
            required dimensions, for which ``hasFull()`` may not return `True`.

        Returns
        -------
        split : `SplitByStateFlags`
            A dataclass holding the indicated data IDs in attributes that
            correspond to the boolean keyword arguments.
        """
        if dataIds is None:
            dataIds = self.allDataIds
        assert dataIds.hasFull() and dataIds.hasRecords()
        result = SplitByStateFlags(expanded=dataIds)
        if complete:
            result.complete = DataCoordinateSequence(
                [DataCoordinate.standardize(e.full.byName(), graph=dataIds.graph) for e in result.expanded],
                graph=dataIds.graph
            )
            self.assertTrue(result.complete.hasFull())
            self.assertFalse(result.complete.hasRecords())
        if minimal:
            result.minimal = DataCoordinateSequence(
                [DataCoordinate.standardize(e.byName(), graph=dataIds.graph) for e in result.expanded],
                graph=dataIds.graph
            )
            self.assertEqual(result.minimal.hasFull(), not dataIds.graph.implied)
            self.assertFalse(result.minimal.hasRecords())
        if not expanded:
            result.expanded = None
        return result

    def testMappingInterface(self):
        """Test that the mapping interface in `DataCoordinate` and (when
        applicable) its ``full`` property are self-consistent and consistent
        with the ``graph`` property.
        """
        for n in range(5):
            dimensions = self.randomDimensionSubset()
            dataIds = self.randomDataIds(n=1).subset(dimensions)
            split = self.splitByStateFlags(dataIds)
            for dataId in split.chain():
                with self.subTest(dataId=dataId):
                    self.assertEqual(list(dataId.values()), [dataId[d] for d in dataId.keys()])
                    self.assertEqual(list(dataId.values()), [dataId[d.name] for d in dataId.keys()])
                    self.assertEqual(dataId.keys(), dataId.graph.required)
            for dataId in itertools.chain(split.complete, split.expanded):
                with self.subTest(dataId=dataId):
                    self.assertTrue(dataId.hasFull())
                    self.assertEqual(dataId.graph.dimensions, dataId.full.keys())
                    self.assertEqual(list(dataId.full.values()), [dataId[k] for k in dataId.graph.dimensions])

    def testEquality(self):
        """Test that different `DataCoordinate` instances with different state
        flags can be compared with each other and other mappings.
        """
        dataIds = self.randomDataIds(n=2)
        split = self.splitByStateFlags(dataIds)
        # Iterate over all combinations of different states of DataCoordinate,
        # with the same underlying data ID values.
        for a0, b0 in itertools.combinations(split.chain(0), 2):
            self.assertEqual(a0, b0)
            self.assertEqual(a0, b0.byName())
            self.assertEqual(a0.byName(), b0)
        # Same thing, for a different data ID value.
        for a1, b1 in itertools.combinations(split.chain(1), 2):
            self.assertEqual(a1, b1)
            self.assertEqual(a1, b1.byName())
            self.assertEqual(a1.byName(), b1)
        # Iterate over all combinations of different states of DataCoordinate,
        # with different underlying data ID values.
        for a0, b1 in itertools.product(split.chain(0), split.chain(1)):
            self.assertNotEqual(a0, b1)
            self.assertNotEqual(a1, b0)
            self.assertNotEqual(a0, b1.byName())
            self.assertNotEqual(a0.byName(), b1)
            self.assertNotEqual(a1, b0.byName())
            self.assertNotEqual(a1.byName(), b0)

    def testStandardize(self):
        """Test constructing a DataCoordinate from many different kinds of
        input via `DataCoordinate.standardize` and `DataCoordinate.subset`.
        """
        for n in range(5):
            dimensions = self.randomDimensionSubset()
            dataIds = self.randomDataIds(n=1).subset(dimensions)
            split = self.splitByStateFlags(dataIds)
            for m, dataId in enumerate(split.chain()):
                # Passing in any kind of DataCoordinate alone just returns
                # that object.
                self.assertIs(dataId, DataCoordinate.standardize(dataId))
                # Same if we also explicitly pass the dimensions we want.
                self.assertIs(dataId, DataCoordinate.standardize(dataId, graph=dataId.graph))
                # Same if we pass the dimensions and some irrelevant
                # kwargs.
                self.assertIs(dataId, DataCoordinate.standardize(dataId, graph=dataId.graph, htm7=12))
                # Test constructing a new data ID from this one with a
                # subset of the dimensions.
                # This is not possible for some combinations of
                # dimensions if hasFull is False (see
                # `DataCoordinate.subset` docs).
                newDimensions = self.randomDimensionSubset(n=1, graph=dataId.graph)
                if dataId.hasFull() or dataId.graph.required.issuperset(newDimensions.required):
                    newDataIds = [
                        dataId.subset(newDimensions),
                        DataCoordinate.standardize(dataId, graph=newDimensions),
                        DataCoordinate.standardize(dataId, graph=newDimensions, htm7=12),
                    ]
                    for newDataId in newDataIds:
                        with self.subTest(newDataId=newDataId, type=type(dataId)):
                            commonKeys = dataId.keys() & newDataId.keys()
                            self.assertTrue(commonKeys)
                            self.assertEqual(
                                [newDataId[k] for k in commonKeys],
                                [dataId[k] for k in commonKeys],
                            )
                            # This should never "downgrade" from
                            # Complete to Minimal or Expanded to Complete.
                            if dataId.hasRecords():
                                self.assertTrue(newDataId.hasRecords())
                            if dataId.hasFull():
                                self.assertTrue(newDataId.hasFull())
            # Start from a complete data ID, and pass its values in via several
            # different ways that should be equivalent.
            for dataId in split.complete:
                # Split the keys (dimension names) into two random subsets, so
                # we can pass some as kwargs below.
                keys1 = set(self.rng.sample(list(dataId.graph.dimensions.names),
                                            len(dataId.graph.dimensions)//2))
                keys2 = dataId.graph.dimensions.names - keys1
                newCompleteDataIds = [
                    DataCoordinate.standardize(dataId.full.byName(), universe=dataId.universe),
                    DataCoordinate.standardize(dataId.full.byName(), graph=dataId.graph),
                    DataCoordinate.standardize(DataCoordinate.makeEmpty(dataId.graph.universe),
                                               **dataId.full.byName()),
                    DataCoordinate.standardize(DataCoordinate.makeEmpty(dataId.graph.universe),
                                               graph=dataId.graph, **dataId.full.byName()),
                    DataCoordinate.standardize(**dataId.full.byName(), universe=dataId.universe),
                    DataCoordinate.standardize(graph=dataId.graph, **dataId.full.byName()),
                    DataCoordinate.standardize(
                        {k: dataId[k] for k in keys1},
                        universe=dataId.universe,
                        **{k: dataId[k] for k in keys2}
                    ),
                    DataCoordinate.standardize(
                        {k: dataId[k] for k in keys1},
                        graph=dataId.graph,
                        **{k: dataId[k] for k in keys2}
                    ),
                ]
                for newDataId in newCompleteDataIds:
                    with self.subTest(dataId=dataId, newDataId=newDataId, type=type(dataId)):
                        self.assertEqual(dataId, newDataId)
                        self.assertTrue(newDataId.hasFull())

    def testRegions(self):
        """Test that data IDs for a few known dimensions have the expected
        regions.
        """
        for dataId in self.randomDataIds(n=4).subset(
                DimensionGraph(self.allDataIds.universe, names=["visit"])):
            self.assertIsNotNone(dataId.region)
            self.assertEqual(dataId.graph.spatial.names, {"visit"})
            self.assertEqual(dataId.region, dataId.records["visit"].region)
        for dataId in self.randomDataIds(n=4).subset(
                DimensionGraph(self.allDataIds.universe, names=["visit", "detector"])):
            self.assertIsNotNone(dataId.region)
            self.assertEqual(dataId.graph.spatial.names, {"visit_detector_region"})
            self.assertEqual(dataId.region, dataId.records["visit_detector_region"].region)
        for dataId in self.randomDataIds(n=4).subset(
                DimensionGraph(self.allDataIds.universe, names=["tract"])):
            self.assertIsNotNone(dataId.region)
            self.assertEqual(dataId.graph.spatial.names, {"tract"})
            self.assertEqual(dataId.region, dataId.records["tract"].region)
        for dataId in self.randomDataIds(n=4).subset(
                DimensionGraph(self.allDataIds.universe, names=["patch"])):
            self.assertIsNotNone(dataId.region)
            self.assertEqual(dataId.graph.spatial.names, {"patch"})
            self.assertEqual(dataId.region, dataId.records["patch"].region)

    def testTimespans(self):
        """Test that data IDs for a few known dimensions have the expected
        timespans.
        """
        for dataId in self.randomDataIds(n=4).subset(
                DimensionGraph(self.allDataIds.universe, names=["visit"])):
            self.assertIsNotNone(dataId.timespan)
            self.assertEqual(dataId.graph.temporal.names, {"visit"})
            self.assertEqual(dataId.timespan, dataId.records["visit"].timespan)

    def testIterableStatusFlags(self):
        """Test that DataCoordinateSet and DataCoordinateSequence compute
        their hasFull and hasRecords flags correctly from their elements.
        """
        dataIds = self.randomDataIds(n=10)
        split = self.splitByStateFlags(dataIds)
        for cls in (DataCoordinateSet, DataCoordinateSequence):
            self.assertTrue(cls(split.expanded, graph=dataIds.graph, check=True).hasFull())
            self.assertTrue(cls(split.expanded, graph=dataIds.graph, check=False).hasFull())
            self.assertTrue(cls(split.expanded, graph=dataIds.graph, check=True).hasRecords())
            self.assertTrue(cls(split.expanded, graph=dataIds.graph, check=False).hasRecords())
            self.assertTrue(cls(split.complete, graph=dataIds.graph, check=True).hasFull())
            self.assertTrue(cls(split.complete, graph=dataIds.graph, check=False).hasFull())
            self.assertFalse(cls(split.complete, graph=dataIds.graph, check=True).hasRecords())
            self.assertFalse(cls(split.complete, graph=dataIds.graph, check=False).hasRecords())
            with self.assertRaises(ValueError):
                cls(split.complete, graph=dataIds.graph, hasRecords=True, check=True)
            self.assertEqual(cls(split.minimal, graph=dataIds.graph, check=True).hasFull(),
                             not dataIds.graph.implied)
            self.assertEqual(cls(split.minimal, graph=dataIds.graph, check=False).hasFull(),
                             not dataIds.graph.implied)
            self.assertFalse(cls(split.minimal, graph=dataIds.graph, check=True).hasRecords())
            self.assertFalse(cls(split.minimal, graph=dataIds.graph, check=False).hasRecords())
            with self.assertRaises(ValueError):
                cls(split.minimal, graph=dataIds.graph, hasRecords=True, check=True)
            if dataIds.graph.implied:
                with self.assertRaises(ValueError):
                    cls(split.minimal, graph=dataIds.graph, hasFull=True, check=True)

    def testSetOperations(self):
        """Test for self-consistency across DataCoordinateSet's operations.
        """
        c = self.randomDataIds(n=10).toSet()
        a = self.randomDataIds(n=20).toSet() | c
        b = self.randomDataIds(n=20).toSet() | c
        # Make sure we don't have a particularly unlucky random seed, since
        # that would make a lot of this test uninteresting.
        self.assertNotEqual(a, b)
        self.assertGreater(len(a), 0)
        self.assertGreater(len(b), 0)
        # The rest of the tests should not depend on the random seed.
        self.assertEqual(a, a)
        self.assertNotEqual(a, a.toSequence())
        self.assertEqual(a, a.toSequence().toSet())
        self.assertEqual(a, a.toSequence().toSet())
        self.assertEqual(b, b)
        self.assertNotEqual(b, b.toSequence())
        self.assertEqual(b, b.toSequence().toSet())
        self.assertEqual(a & b, a.intersection(b))
        self.assertLessEqual(a & b, a)
        self.assertLessEqual(a & b, b)
        self.assertEqual(a | b, a.union(b))
        self.assertGreaterEqual(a | b, a)
        self.assertGreaterEqual(a | b, b)
        self.assertEqual(a - b, a.difference(b))
        self.assertLessEqual(a - b, a)
        self.assertLessEqual(b - a, b)
        self.assertEqual(a ^ b, a.symmetric_difference(b))
        self.assertGreaterEqual(a ^ b, (a | b) - (a & b))


if __name__ == "__main__":
    unittest.main()
