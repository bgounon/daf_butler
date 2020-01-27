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

"""Unit tests for `lsst.daf.butler.tests.testRepo`, a module for creating
test repositories or butlers.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from lsst.daf.butler.tests import makeTestRepo, makeTestCollection, addDatasetType, expandUniqueId


TESTDIR = os.path.abspath(os.path.dirname(__file__))


class ButlerUtilsTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Repository should be re-created for each test case, but
        # this has a prohibitive run-time cost at present
        cls.root = tempfile.mkdtemp(dir=TESTDIR)

        dataIds = {
            "instrument": ["notACam", "dummyCam"],
            "physical_filter": ["k2020", "l2019"],
            "visit": [101, 102],
            "detector": [5]
        }
        cls.creatorButler = makeTestRepo(cls.root, dataIds)

        addDatasetType(cls.creatorButler, "DataType1", {"instrument"}, "NumpyArray")
        addDatasetType(cls.creatorButler, "DataType2", {"instrument", "visit", "detector"}, "NumpyArray")

    @classmethod
    def tearDownClass(cls):
        # TODO: use addClassCleanup rather than tearDownClass in Python 3.8
        # to keep the addition and removal together and make it more robust
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.butler = makeTestCollection(self.creatorButler)

    def testButlerValid(self):
        self.butler.validateConfiguration()

    def _checkButlerDimension(self, dimensions, query, expected):
        result = [id for id in self.butler.registry.queryDimensions(
            dimensions,
            where=query,
            expand=False)]
        self.assertEqual(len(result), 1)
        self.assertIn(dict(result[0]), expected)

    def testButlerDimensions(self):
        self. _checkButlerDimension({"instrument"},
                                    "instrument='notACam'",
                                    [{"instrument": "notACam"}, {"instrument": "dummyCam"}])
        self. _checkButlerDimension({"visit", "instrument"},
                                    "visit=101",
                                    [{"instrument": "notACam", "visit": 101},
                                     {"instrument": "dummyCam", "visit": 101}])
        self. _checkButlerDimension({"visit", "instrument"},
                                    "visit=102",
                                    [{"instrument": "notACam", "visit": 102},
                                     {"instrument": "dummyCam", "visit": 102}])
        self. _checkButlerDimension({"detector", "instrument"},
                                    "detector=5",
                                    [{"instrument": "notACam", "detector": 5},
                                     {"instrument": "dummyCam", "detector": 5}])

    def testAddDatasetType(self):
        self.assertEqual(len(self.butler.registry.getAllDatasetTypes()), 2)

        # Testing the DatasetType objects is not practical, because all tests
        # need a DimensionUniverse. So just check that we have the dataset
        # types we expect.
        self.butler.registry.getDatasetType("DataType1")
        self.butler.registry.getDatasetType("DataType2")

        with self.assertRaises(ValueError):
            addDatasetType(self.butler, "DataType3", {"4thDimension"}, "NumpyArray")
        with self.assertRaises(ValueError):
            addDatasetType(self.butler, "DataType3", {"instrument"}, "UnstorableType")

    def testUniqueButler(self):
        dataId = {"instrument": "notACam"}
        self.butler.put(np.array([1, 2, 3]), "DataType1", dataId)
        self.assertTrue(self.butler.datasetExists("DataType1", dataId))

        newButler = makeTestCollection(self.creatorButler)
        with self.assertRaises(LookupError):
            newButler.datasetExists("DataType1", dataId)

    def testExpandUniqueId(self):
        self.assertEqual(dict(expandUniqueId(self.butler, {"instrument": "notACam"})),
                         {"instrument": "notACam"})
        self.assertIn(dict(expandUniqueId(self.butler, {"visit": 101})),
                      [{"instrument": "notACam", "visit": 101},
                       {"instrument": "dummyCam", "visit": 101}])
        self.assertIn(dict(expandUniqueId(self.butler, {"detector": 5})),
                      [{"instrument": "notACam", "detector": 5},
                       {"instrument": "dummyCam", "detector": 5}])
        self.assertIn(dict(expandUniqueId(self.butler, {"physical_filter": "k2020"})),
                      [{"instrument": "notACam", "physical_filter": "k2020"},
                       {"instrument": "notACam", "physical_filter": "k2020"}])
        with self.assertRaises(ValueError):
            expandUniqueId(self.butler, {"tract": 42})


if __name__ == "__main__":
    unittest.main()
