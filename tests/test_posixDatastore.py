#
# LSST Data Management System
#
# Copyright 2008-2017  AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import os
import unittest

import lsst.utils.tests
import lsst.afw.table

from lsst.butler.datastore import PosixDatastore
from lsst.butler.storageClass import SourceCatalog


class PosixDatastoreTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        testDir = os.path.dirname(__file__)
        catalogPath = os.path.join(testDir, "data", "basic", "source_catalog.fits")
        self.catalog = lsst.afw.table.SourceCatalog.readFits(catalogPath)

    def _assertCatalogEqual(self, inputCatalog, outputCatalog):
        self.assertIsInstance(outputCatalog, lsst.afw.table.SourceCatalog)

    def testConstructor(self):
        datastore = PosixDatastore()

    def testBasicPutGet(self):
        datastore = PosixDatastore()
        # Put
        storageClass = SourceCatalog
        uri, _ = datastore.put(self.catalog, storageClass=storageClass, path="tester.fits", typeName=None)
        # Get
        out = datastore.get(uri, storageClass=storageClass, parameters=None)
        self._assertCatalogEqual(self.catalog, out)
        # These should raise
        with self.assertRaises(ValueError):
            # non-existing file
            datastore.get(uri="file:///non_existing.fits", storageClass=storageClass, parameters=None)
        with self.assertRaises(ValueError):
            # invalid storage class
            datastore.get(uri="file:///non_existing.fits", storageClass=object, parameters=None)

    def testRemove(self):
        datastore = PosixDatastore()
        # Put
        storageClass = SourceCatalog
        uri, _ = datastore.put(self.catalog, storageClass=storageClass, path="tester.fits", typeName=None)
        # Get
        out = datastore.get(uri, storageClass=storageClass, parameters=None)
        self._assertCatalogEqual(self.catalog, out)
        # Remove
        datastore.remove(uri)
        # Get should now fail
        with self.assertRaises(ValueError):
            datastore.get(uri, storageClass=storageClass, parameters=None)
        # Can only delete once
        with self.assertRaises(ValueError):
            datastore.remove(uri)

    def testTransfer(self):
        path = "tester.fits"
        inputPosixDatastore = PosixDatastore("test_input_datastore", create=True)
        outputPosixDatastore = PosixDatastore("test_output_datastore", create=True)
        storageClass = SourceCatalog
        inputUri, _ = inputPosixDatastore.put(self.catalog, storageClass, path)
        outputUri, _ = outputPosixDatastore.transfer(inputPosixDatastore, inputUri, storageClass, path)
        outCatalog = outputPosixDatastore.get(outputUri, storageClass)
        self._assertCatalogEqual(self.catalog, outCatalog)

    def testConfig(self):
        datastore = PosixDatastore()
        datastore.config.dumpToFile("posix_datastore_config.yaml")
        datastore.config.loadFromFile("posix_datastore_config.yaml")


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
