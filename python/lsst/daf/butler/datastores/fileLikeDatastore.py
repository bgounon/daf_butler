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

"""Generic file-based datastore code."""

__all__ = ("FileLikeDatastore", )

import logging
from abc import abstractmethod

from sqlalchemy import BigInteger, String

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from lsst.daf.butler import (
    ButlerURI,
    CompositesMap,
    Config,
    FileDataset,
    DatasetRef,
    DatasetType,
    DatasetTypeNotSupportedError,
    Datastore,
    DatastoreConfig,
    DatastoreValidationError,
    FileDescriptor,
    FileTemplates,
    FileTemplateValidationError,
    Formatter,
    FormatterFactory,
    Location,
    LocationFactory,
    StorageClass,
    StoredFileInfo,
)

from lsst.daf.butler import ddl
from lsst.daf.butler.registry.interfaces import (
    ReadOnlyDatabaseError,
    DatastoreRegistryBridge,
)

from lsst.daf.butler.core.repoRelocation import replaceRoot
from lsst.daf.butler.core.utils import getInstanceOf, getClassOf, transactional
from .genericDatastore import GenericBaseDatastore

if TYPE_CHECKING:
    from lsst.daf.butler import LookupKey
    from lsst.daf.butler.registry.interfaces import DatasetIdRef, DatastoreRegistryBridgeManager

log = logging.getLogger(__name__)

# String to use when a Python None is encountered
NULLSTR = "__NULL_STRING__"


class _IngestPrepData(Datastore.IngestPrepData):
    """Helper class for FileLikeDatastore ingest implementation.

    Parameters
    ----------
    datasets : `list` of `FileDataset`
        Files to be ingested by this datastore.
    """
    def __init__(self, datasets: List[FileDataset]):
        super().__init__(ref for dataset in datasets for ref in dataset.refs)
        self.datasets = datasets


@dataclass(frozen=True)
class DatastoreFileGetInformation:
    """Collection of useful parameters needed to retrieve a file from
    a Datastore.
    """

    location: Location
    """The location from which to read the dataset."""

    formatter: Formatter
    """The `Formatter` to use to deserialize the dataset."""

    info: StoredFileInfo
    """Stored information about this file and its formatter."""

    assemblerParams: dict
    """Parameters to use for post-processing the retrieved dataset."""

    component: Optional[str]
    """The component to be retrieved (can be `None`)."""

    readStorageClass: StorageClass
    """The `StorageClass` of the dataset being read."""


class FileLikeDatastore(GenericBaseDatastore):
    """Generic Datastore for file-based implementations.

    Should always be sub-classed since key abstract methods are missing.

    Parameters
    ----------
    config : `DatastoreConfig` or `str`
        Configuration as either a `Config` object or URI to file.
    bridgeManager : `DatastoreRegistryBridgeManager`
        Object that manages the interface between `Registry` and datastores.
    butlerRoot : `str`, optional
        New datastore root to use to override the configuration value.

    Raises
    ------
    ValueError
        If root location does not exist and ``create`` is `False` in the
        configuration.
    """

    defaultConfigFile: ClassVar[Optional[str]] = None
    """Path to configuration defaults. Relative to $DAF_BUTLER_DIR/config or
    absolute path. Can be None if no defaults specified.
    """

    root: str
    """Root directory or URI of this `Datastore`."""

    locationFactory: LocationFactory
    """Factory for creating locations relative to the datastore root."""

    formatterFactory: FormatterFactory
    """Factory for creating instances of formatters."""

    templates: FileTemplates
    """File templates that can be used by this `Datastore`."""

    composites: CompositesMap
    """Determines whether a dataset should be disassembled on put."""

    @classmethod
    def setConfigRoot(cls, root: str, config: Config, full: Config, overwrite: bool = True) -> None:
        """Set any filesystem-dependent config options for this Datastore to
        be appropriate for a new empty repository with the given root.

        Parameters
        ----------
        root : `str`
            URI to the root of the data repository.
        config : `Config`
            A `Config` to update. Only the subset understood by
            this component will be updated. Will not expand
            defaults.
        full : `Config`
            A complete config with all defaults expanded that can be
            converted to a `DatastoreConfig`. Read-only and will not be
            modified by this method.
            Repository-specific options that should not be obtained
            from defaults when Butler instances are constructed
            should be copied from ``full`` to ``config``.
        overwrite : `bool`, optional
            If `False`, do not modify a value in ``config`` if the value
            already exists.  Default is always to overwrite with the provided
            ``root``.

        Notes
        -----
        If a keyword is explicitly defined in the supplied ``config`` it
        will not be overridden by this method if ``overwrite`` is `False`.
        This allows explicit values set in external configs to be retained.
        """
        Config.updateParameters(DatastoreConfig, config, full,
                                toUpdate={"root": root},
                                toCopy=("cls", ("records", "table")), overwrite=overwrite)

    @classmethod
    def makeTableSpec(cls) -> ddl.TableSpec:
        return ddl.TableSpec(
            fields=[
                ddl.FieldSpec(name="dataset_id", dtype=BigInteger, primaryKey=True),
                ddl.FieldSpec(name="path", dtype=String, length=256, nullable=False),
                ddl.FieldSpec(name="formatter", dtype=String, length=128, nullable=False),
                ddl.FieldSpec(name="storage_class", dtype=String, length=64, nullable=False),
                # Use empty string to indicate no component
                ddl.FieldSpec(name="component", dtype=String, length=32, primaryKey=True),
                # TODO: should checksum be Base64Bytes instead?
                ddl.FieldSpec(name="checksum", dtype=String, length=128, nullable=True),
                ddl.FieldSpec(name="file_size", dtype=BigInteger, nullable=True),
            ],
            unique=frozenset(),
        )

    def __init__(self, config: Union[DatastoreConfig, str],
                 bridgeManager: DatastoreRegistryBridgeManager, butlerRoot: str = None):
        super().__init__(config, bridgeManager)
        if "root" not in self.config:
            raise ValueError("No root directory specified in configuration")

        # Name ourselves either using an explicit name or a name
        # derived from the (unexpanded) root
        if "name" in self.config:
            self.name = self.config["name"]
        else:
            # We use the unexpanded root in the name to indicate that this
            # datastore can be moved without having to update registry.
            self.name = "{}@{}".format(type(self).__name__,
                                       self.config["root"])

        # Support repository relocation in config
        # Existence of self.root is checked in subclass
        self.root = replaceRoot(self.config["root"], butlerRoot)

        self.locationFactory = LocationFactory(self.root)
        self.formatterFactory = FormatterFactory()

        # Now associate formatters with storage classes
        self.formatterFactory.registerFormatters(self.config["formatters"],
                                                 universe=bridgeManager.universe)

        # Read the file naming templates
        self.templates = FileTemplates(self.config["templates"],
                                       universe=bridgeManager.universe)

        # See if composites should be disassembled
        self.composites = CompositesMap(self.config["composites"],
                                        universe=bridgeManager.universe)

        tableName = self.config["records", "table"]
        try:
            # Storage of paths and formatters, keyed by dataset_id
            self._table = bridgeManager.opaque.register(tableName, self.makeTableSpec())
            # Interface to Registry.
            self._bridge = bridgeManager.register(self.name)
        except ReadOnlyDatabaseError:
            # If the database is read only and we just tried and failed to
            # create a table, it means someone is trying to create a read-only
            # butler client for an empty repo.  That should be okay, as long
            # as they then try to get any datasets before some other client
            # creates the table.  Chances are they'rejust validating
            # configuration.
            pass

        # Determine whether checksums should be used
        self.useChecksum = self.config.get("checksum", True)

    def __str__(self) -> str:
        return self.root

    @property
    def bridge(self) -> DatastoreRegistryBridge:
        return self._bridge

    @abstractmethod
    def _artifact_exists(self, location: Location) -> bool:
        """Check that an artifact exists in this datastore at the specified
        location.

        Parameters
        ----------
        location : `Location`
            Expected location of the artifact associated with this datastore.

        Returns
        -------
        exists : `bool`
            True if the location can be found, false otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def _delete_artifact(self, location: Location) -> None:
        """Delete the artifact from the datastore.

        Parameters
        ----------
        location : `Location`
            Location of the artifact associated with this datastore.
        """
        raise NotImplementedError()

    def addStoredItemInfo(self, refs: Iterable[DatasetRef], infos: Iterable[StoredFileInfo]) -> None:
        # Docstring inherited from GenericBaseDatastore
        records = []
        foundNull = 0
        for ref, info in zip(refs, infos):
            # Component should come from ref and fall back on info
            component = ref.datasetType.component()
            if component is None and info.component is not None:
                component = info.component

            # Can not have component not None and path None
            # since that would imply we had no artifact for this component
            if component is not None and info.path is None:
                raise RuntimeError(f"Internal error in datastore {self.name}:"
                                   f" Component is {component} but no path for artifact")

            # Check that we only have one null entry with no component and
            # no path
            if component is None and info.path is None:
                foundNull += 1

            if component is None:
                # Use empty string since we want this to be part of the
                # primary key.
                component = NULLSTR
            path = info.path if info.path is not None else NULLSTR
            records.append(
                dict(dataset_id=ref.id, formatter=info.formatter, path=path,
                     storage_class=info.storageClass.name, component=component,
                     checksum=info.checksum, file_size=info.file_size)
            )

        if foundNull > 1 or (len(refs) == 1 and foundNull > 0):
            raise RuntimeError(f"Internal error in datastore {self.name}"
                               " Component and path are None in too many records "
                               f"(nulls: {foundNull} / records: {len(refs)}).")

        self._table.insert(*records)

    def getStoredItemsInfo(self, ref: DatasetIdRef) -> List[StoredFileInfo]:
        # Docstring inherited from GenericBaseDatastore

        # Look for the dataset_id -- there might be multiple matches
        # if we have disassembled the dataset.
        records = list(self._table.fetch(dataset_id=ref.id))

        results = []
        for record in records:
            # Convert name of StorageClass to instance
            storageClass = self.storageClassFactory.getStorageClass(record["storage_class"])
            component = record["component"] if (record["component"]
                                                and record["component"] != NULLSTR) else None
            path = None if record["path"] == NULLSTR else record["path"]

            if path is None and component is not None:
                raise RuntimeError(f"Internal error in datastore {self.name} reading "
                                   f"component {component} information from registry.")

            info = StoredFileInfo(formatter=record["formatter"],
                                  path=path,
                                  storageClass=storageClass,
                                  component=component,
                                  checksum=record["checksum"],
                                  file_size=record["file_size"])
            results.append(info)

        return results

    def _registered_refs_per_artifact(self, pathInStore: str) -> Set[int]:
        """Return all dataset refs associated with the supplied path.

        Parameters
        ----------
        pathInStore : `str`
            Path of interest in the data store.

        Returns
        -------
        ids : `set` of `int`
            All `DatasetRef` IDs associated with this path.
        """
        records = list(self._table.fetch(path=pathInStore))
        ids = {r["dataset_id"] for r in records}
        return ids

    def removeStoredItemInfo(self, ref: DatasetIdRef) -> None:
        # Docstring inherited from GenericBaseDatastore
        self._table.delete(dataset_id=ref.id)

    def _get_dataset_locations_info(self, ref: DatasetIdRef) -> List[Tuple[Location, StoredFileInfo]]:
        r"""Find all the `Location`\ s  of the requested dataset in the
        `Datastore` and the associated stored file information.

        Parameters
        ----------
        ref : `DatasetRef`
            Reference to the required `Dataset`.

        Returns
        -------
        results : `list` [`tuple` [`Location`, `StoredFileInfo` ]]
            Location of the dataset within the datastore and
            stored information about each file and its formatter.
        """
        # Get the file information (this will fail if no file)
        records = self.getStoredItemsInfo(ref)

        # Use the path to determine the location
        return [(self.locationFactory.fromPath(r.path) if r.path is not None else None, r) for r in records]

    def _can_remove_dataset_artifact(self, ref: DatasetIdRef, location: Location) -> bool:
        """Check that there is only one dataset associated with the
        specified artifact.

        Parameters
        ----------
        ref : `DatasetRef` or `FakeDatasetRef`
            Dataset to be removed.
        location : `Location`
            The location of the artifact to be removed.

        Returns
        -------
        can_remove : `Bool`
            True if the artifact can be safely removed.
        """

        # Get all entries associated with this path
        allRefs = self._registered_refs_per_artifact(location.pathInStore)
        if not allRefs:
            raise RuntimeError(f"Datastore inconsistency error. {location.pathInStore} not in registry")

        # Remove these refs from all the refs and if there is nothing left
        # then we can delete
        remainingRefs = allRefs - {ref.id}

        if remainingRefs:
            return False
        return True

    def _prepare_for_get(self, ref: DatasetRef,
                         parameters: Optional[Mapping[str, Any]] = None) -> List[DatastoreFileGetInformation]:
        """Check parameters for ``get`` and obtain formatter and
        location.

        Parameters
        ----------
        ref : `DatasetRef`
            Reference to the required Dataset.
        parameters : `dict`
            `StorageClass`-specific parameters that specify, for example,
            a slice of the dataset to be loaded.

        Returns
        -------
        getInfo : `list` [`DatastoreFileGetInformation`]
            Parameters needed to retrieve each file.
        """
        log.debug("Retrieve %s from %s with parameters %s", ref, self.name, parameters)

        # Get file metadata and internal metadata
        fileLocations = self._get_dataset_locations_info(ref)
        if not fileLocations:
            raise FileNotFoundError(f"Could not retrieve dataset {ref}.")

        # The storage class we want to use eventually
        refStorageClass = ref.datasetType.storageClass

        # Check that the supplied parameters are suitable for the type read
        refStorageClass.validateParameters(parameters)

        if len(fileLocations) > 1:
            disassembled = True
        else:
            disassembled = False

        # Is this a component request?
        refComponent = ref.datasetType.component()

        fileGetInfo = []
        for location, storedFileInfo in fileLocations:

            # The storage class used to write the file
            writeStorageClass = storedFileInfo.storageClass

            # If this has been disassembled we need read to match the write
            if disassembled:
                readStorageClass = writeStorageClass
            else:
                readStorageClass = refStorageClass

            formatter = getInstanceOf(storedFileInfo.formatter,
                                      FileDescriptor(location, readStorageClass=readStorageClass,
                                                     storageClass=writeStorageClass, parameters=parameters),
                                      ref.dataId)

            _, notFormatterParams = formatter.segregateParameters()

            # Of the remaining parameters, extract the ones supported by
            # this StorageClass (for components not all will be handled)
            assemblerParams = readStorageClass.filterParameters(notFormatterParams)

            # The ref itself could be a component if the dataset was
            # disassembled by butler, or we disassembled in datastore and
            # components came from the datastore records
            # We should not set component for a null entry that has a missing
            # path
            if location is None:
                component = None
            else:
                component = storedFileInfo.component if storedFileInfo.component else refComponent

            fileGetInfo.append(DatastoreFileGetInformation(location, formatter, storedFileInfo,
                                                           assemblerParams, component, readStorageClass))

        return fileGetInfo

    def _prepare_for_put(self, inMemoryDataset: Any, ref: DatasetRef) -> Tuple[Location, Formatter]:
        """Check the arguments for ``put`` and obtain formatter and
        location.

        Parameters
        ----------
        inMemoryDataset : `object`
            The dataset to store.
        ref : `DatasetRef`
            Reference to the associated Dataset.

        Returns
        -------
        location : `Location`
            The location to write the dataset.
        formatter : `Formatter`
            The `Formatter` to use to write the dataset.

        Raises
        ------
        TypeError
            Supplied object and storage class are inconsistent.
        DatasetTypeNotSupportedError
            The associated `DatasetType` is not handled by this datastore.
        """
        self._validate_put_parameters(inMemoryDataset, ref)

        # Work out output file name
        try:
            template = self.templates.getTemplate(ref)
        except KeyError as e:
            raise DatasetTypeNotSupportedError(f"Unable to find template for {ref}") from e

        location = self.locationFactory.fromPath(template.format(ref))

        # Get the formatter based on the storage class
        storageClass = ref.datasetType.storageClass
        try:
            formatter = self.formatterFactory.getFormatter(ref,
                                                           FileDescriptor(location,
                                                                          storageClass=storageClass),
                                                           ref.dataId)
        except KeyError as e:
            raise DatasetTypeNotSupportedError(f"Unable to find formatter for {ref}") from e

        # Now that we know the formatter, update the location
        location = formatter.makeUpdatedLocation(location)

        return location, formatter

    @abstractmethod
    def _standardizeIngestPath(self, path: str, *, transfer: Optional[str] = None) -> str:
        """Standardize the path of a to-be-ingested file.

        Parameters
        ----------
        path : `str`
            Path of a file to be ingested.
        transfer : `str`, optional
            How (and whether) the dataset should be added to the datastore.
            See `ingest` for details of transfer modes.
            This implementation is provided only so
            `NotImplementedError` can be raised if the mode is not supported;
            actual transfers are deferred to `_extractIngestInfo`.

        Returns
        -------
        path : `str`
            New path in what the datastore considers standard form.

        Notes
        -----
        Subclasses of `FileLikeDatastore` should implement this method instead
        of `_prepIngest`.  It should not modify the data repository or given
        file in any way.

        Raises
        ------
        NotImplementedError
            Raised if the datastore does not support the given transfer mode
            (including the case where ingest is not supported at all).
        FileNotFoundError
            Raised if one of the given files does not exist.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    @abstractmethod
    def _extractIngestInfo(self, path: str, ref: DatasetRef, *,
                           formatter: Union[Formatter, Type[Formatter]],
                           transfer: Optional[str] = None) -> StoredFileInfo:
        """Relocate (if necessary) and extract `StoredFileInfo` from a
        to-be-ingested file.

        Parameters
        ----------
        path : `str`
            Path of a file to be ingested.
        ref : `DatasetRef`
            Reference for the dataset being ingested.  Guaranteed to have
            ``dataset_id not None`.
        formatter : `type` or `Formatter`
            `Formatter` subclass to use for this dataset or an instance.
        transfer : `str`, optional
            How (and whether) the dataset should be added to the datastore.
            See `ingest` for details of transfer modes.

        Returns
        -------
        info : `StoredFileInfo`
            Internal datastore record for this file.  This will be inserted by
            the caller; the `_extractIngestInfo` is only resposible for
            creating and populating the struct.

        Raises
        ------
        FileNotFoundError
            Raised if one of the given files does not exist.
        FileExistsError
            Raised if transfer is not `None` but the (internal) location the
            file would be moved to is already occupied.
        """
        raise NotImplementedError("Must be implemented by subclasses.")

    def _prepIngest(self, *datasets: FileDataset, transfer: Optional[str] = None) -> _IngestPrepData:
        # Docstring inherited from Datastore._prepIngest.
        filtered = []
        for dataset in datasets:
            acceptable = [ref for ref in dataset.refs if self.constraints.isAcceptable(ref)]
            if not acceptable:
                continue
            else:
                dataset.refs = acceptable
            if dataset.formatter is None:
                dataset.formatter = self.formatterFactory.getFormatterClass(dataset.refs[0])
            else:
                assert isinstance(dataset.formatter, (type, str))
                dataset.formatter = getClassOf(dataset.formatter)
            dataset.path = self._standardizeIngestPath(dataset.path, transfer=transfer)
            filtered.append(dataset)
        return _IngestPrepData(filtered)

    @transactional
    def _finishIngest(self, prepData: Datastore.IngestPrepData, *, transfer: Optional[str] = None) -> None:
        # Docstring inherited from Datastore._finishIngest.
        refsAndInfos = []
        for dataset in prepData.datasets:
            # Do ingest as if the first dataset ref is associated with the file
            info = self._extractIngestInfo(dataset.path, dataset.refs[0], formatter=dataset.formatter,
                                           transfer=transfer)
            refsAndInfos.extend([(ref, info) for ref in dataset.refs])
        self._register_datasets(refsAndInfos)

    def _calculate_ingested_datastore_name(self, srcUri: ButlerURI, ref: DatasetRef,
                                           formatter: Union[Formatter, Type[Formatter]]) -> Location:
        """Given a source URI and a DatasetRef, determine the name the
        dataset will have inside datastore.

        Parameters
        ----------
        srcUri : `ButlerURI`
            URI to the source dataset file.
        ref : `DatasetRef`
            Ref associated with the newly-ingested dataset artifact.  This
            is used to determine the name within the datastore.
        formatter : `Formatter` or Formatter class.
            Formatter to use for validation. Can be a class or an instance.

        Returns
        -------
        location : `Location`
            Target location for the newly-ingested dataset.
        """
        # Ingesting a file from outside the datastore.
        # This involves a new name.
        template = self.templates.getTemplate(ref)
        location = self.locationFactory.fromPath(template.format(ref))

        # Get the extension
        ext = srcUri.getExtension()

        # Update the destination to include that extension
        location.updateExtension(ext)

        # Ask the formatter to validate this extension
        formatter.validateExtension(location)

        return location

    @abstractmethod
    def _write_in_memory_to_artifact(self, inMemoryDataset: Any, ref: DatasetRef) -> StoredFileInfo:
        """Write out in memory dataset to datastore.

        Parameters
        ----------
        inMemoryDataset : `object`
            Dataset to write to datastore.
        ref : `DatasetRef`
            Registry information associated with this dataset.

        Returns
        -------
        info : `StoredFileInfo`
            Information describin the artifact written to the datastore.
        """
        raise NotImplementedError()

    @abstractmethod
    def _read_artifact_into_memory(self, getInfo: DatastoreFileGetInformation,
                                   ref: DatasetRef, isComponent: bool = False) -> Any:
        """Read the artifact from datastore into in memory object.

        Parameters
        ----------
        getInfo : `DatastoreFileGetInformation`
            Information about the artifact within the datastore.
        ref : `DatasetRef`
            The registry information associated with this artifact.
        isComponent : `bool`
            Flag to indicate if a component is being read from this artifact.

        Returns
        -------
        inMemoryDataset : `object`
            The artifact as a python object.
        """
        raise NotImplementedError()

    def exists(self, ref: DatasetRef) -> bool:
        """Check if the dataset exists in the datastore.

        Parameters
        ----------
        ref : `DatasetRef`
            Reference to the required dataset.

        Returns
        -------
        exists : `bool`
            `True` if the entity exists in the `Datastore`.
        """
        fileLocations = self._get_dataset_locations_info(ref)
        if not fileLocations:
            return False
        for location, storedFileInfo in fileLocations:
            if storedFileInfo.path is None:
                # Composite formatter
                continue
            if not self._artifact_exists(location):
                return False

        return True

    def getURIs(self, ref: DatasetRef,
                predict: bool = False) -> Tuple[Optional[ButlerURI], Dict[str, ButlerURI]]:
        """Return URIs associated with dataset.

        Parameters
        ----------
        ref : `DatasetRef`
            Reference to the required dataset.
        predict : `bool`, optional
            If the datastore does not know about the dataset, should it
            return a predicted URI or not?

        Returns
        -------
        primary : `ButlerURI`
            The URI to the primary artifact associated with this dataset.
            If the dataset was disassembled within the datastore this
            may be `None`.
        components : `dict`
            URIs to any components associated with the dataset artifact.
            Can be empty if there are no components.
        """

        primary: Optional[ButlerURI] = None
        components: Dict[str, ButlerURI] = {}

        # if this has never been written then we have to guess
        if not self.exists(ref):
            if not predict:
                raise FileNotFoundError("Dataset {} not in this datastore".format(ref))

            def predictLocation(thisRef: DatasetRef) -> Location:
                template = self.templates.getTemplate(thisRef)
                location = self.locationFactory.fromPath(template.format(thisRef))
                storageClass = ref.datasetType.storageClass
                formatter = self.formatterFactory.getFormatter(thisRef,
                                                               FileDescriptor(location,
                                                                              storageClass=storageClass))
                # Try to use the extension attribute but ignore problems if the
                # formatter does not define one.
                try:
                    location = formatter.makeUpdatedLocation(location)
                except Exception:
                    # Use the default extension
                    pass
                return location

            doDisassembly = self.composites.shouldBeDisassembled(ref)

            if doDisassembly:

                for component, componentStorage in ref.datasetType.storageClass.components.items():
                    compRef = ref.makeComponentRef(component)
                    compLocation = predictLocation(compRef)

                    # Add a URI fragment to indicate this is a guess
                    components[component] = ButlerURI(compLocation.uri + "#predicted")

            else:

                location = predictLocation(ref)

                # Add a URI fragment to indicate this is a guess
                primary = ButlerURI(location.uri + "#predicted")

            return primary, components

        # If this is a ref that we have written we can get the path.
        # Get file metadata and internal metadata
        fileLocations = self._get_dataset_locations_info(ref)

        if not fileLocations:
            raise RuntimeError(f"Unexpectedly got no artifacts for dataset {ref}")

        if len(fileLocations) == 1:
            # No disassembly so this is the primary URI
            primary = ButlerURI(fileLocations[0][0].uri)

        else:
            for location, storedFileInfo in fileLocations:
                if storedFileInfo.path is None:
                    # composite formatter
                    continue
                if storedFileInfo.component is None:
                    raise RuntimeError(f"Unexpectedly got no component name for a component at {location}")
                components[storedFileInfo.component] = ButlerURI(location.uri)

        return primary, components

    def getURI(self, ref: DatasetRef, predict: bool = False) -> ButlerURI:
        """URI to the Dataset.

        Parameters
        ----------
        ref : `DatasetRef`
            Reference to the required Dataset.
        predict : `bool`
            If `True`, allow URIs to be returned of datasets that have not
            been written.

        Returns
        -------
        uri : `str`
            URI pointing to the dataset within the datastore. If the
            dataset does not exist in the datastore, and if ``predict`` is
            `True`, the URI will be a prediction and will include a URI
            fragment "#predicted".
            If the datastore does not have entities that relate well
            to the concept of a URI the returned URI will be
            descriptive. The returned URI is not guaranteed to be obtainable.

        Raises
        ------
        FileNotFoundError
            Raised if a URI has been requested for a dataset that does not
            exist and guessing is not allowed.
        RuntimeError
            Raised if a request is made for a single URI but multiple URIs
            are associated with this dataset.

        Notes
        -----
        When a predicted URI is requested an attempt will be made to form
        a reasonable URI based on file templates and the expected formatter.
        """
        primary, components = self.getURIs(ref, predict)
        if primary is None or components:
            raise RuntimeError(f"Dataset ({ref}) includes distinct URIs for components. "
                               "Use Dataastore.getURIs() instead.")
        return primary

    def get(self, ref: DatasetRef, parameters: Optional[Mapping[str, Any]] = None) -> Any:
        """Load an InMemoryDataset from the store.

        Parameters
        ----------
        ref : `DatasetRef`
            Reference to the required Dataset.
        parameters : `dict`
            `StorageClass`-specific parameters that specify, for example,
            a slice of the dataset to be loaded.

        Returns
        -------
        inMemoryDataset : `object`
            Requested dataset or slice thereof as an InMemoryDataset.

        Raises
        ------
        FileNotFoundError
            Requested dataset can not be retrieved.
        TypeError
            Return value from formatter has unexpected type.
        ValueError
            Formatter failed to process the dataset.
        """
        allGetInfo = self._prepare_for_get(ref, parameters)
        refComponent = ref.datasetType.component()

        # Supplied storage class for validation
        refStorageClass = ref.datasetType.storageClass

        # Create mapping from component name to related info
        allComponents = {i.component: i for i in allGetInfo}

        isDisassembled = len(allGetInfo) > 1

        # Look for the special case where we are disassembled but the
        # component is a read-only component that was not written during
        # disassembly. For this scenario we need to check that the
        # component requested is listed as a read-only component for the
        # composite storage class
        isDisassembledReadOnlyComponent = False
        if isDisassembled and refComponent:
            # If we are looking for a composite storage class and this
            # composite was disassembled, we can only do this check if
            # the datastore recorded a composite storage class on put.
            # If it did not find one, we can not check so assume it is
            # a normal component.
            if None in allComponents:
                compositeStorageClass = allComponents[None].info.storageClass
                isDisassembledReadOnlyComponent = refComponent in compositeStorageClass.readComponents
            else:
                isDisassembledReadOnlyComponent = False

        if isDisassembled and not refComponent:

            # This was a disassembled dataset spread over multiple files
            # and we need to put them all back together again.
            # Read into memory and then assemble
            usedParams = set()
            components: Dict[str, Any] = {}
            for getInfo in allGetInfo:
                # assemblerParams are parameters not understood by the
                # associated formatter.
                usedParams.update(set(getInfo.assemblerParams))

                component = getInfo.component

                if component is None and getInfo.location is None:
                    # This is the composite formatter
                    continue

                if component is None:
                    raise RuntimeError(f"Internal error in datastore assembly of {ref}")

                # We do not want the formatter to think it's reading
                # a component though because it is really reading a
                # standalone dataset -- always tell reader it is not a
                # component.
                components[component] = self._read_artifact_into_memory(getInfo, ref, isComponent=False)

            inMemoryDataset = ref.datasetType.storageClass.assembler().assemble(components)

            # Any unused parameters will have to be passed to the assembler
            if parameters:
                unusedParams = {k: v for k, v in parameters.items() if k not in usedParams}
            else:
                unusedParams = {}

            # Process parameters
            return ref.datasetType.storageClass.assembler().handleParameters(inMemoryDataset,
                                                                             parameters=unusedParams)
        elif isDisassembledReadOnlyComponent:
            compositeInfo = allComponents.get(None)
            if compositeInfo is None:
                raise RuntimeError(f"Unable to retrieve read-only component '{refComponent}' since"
                                   "no formatter was stored with the composite during disassembly.")
            compositeFormatter = compositeInfo.formatter

            # Assume that every read-only component can be calculated by
            # forwarding the request to a single read/write component.
            # Rather than guessing which rw component is the right one by
            # scanning each for a read-only component of the same name,
            # we ask the composite formatter directly which one is best to
            # use.
            if refComponent is None:
                # Mainly for mypy
                raise RuntimeError(f"Internal error in datastore {self.name}: component can not be None here")
            forwardedComponent = compositeFormatter.selectResponsibleComponent(refComponent,
                                                                               set(allComponents))

            # Select the relevant component
            rwInfo = allComponents[forwardedComponent]

            # Unfortunately the FileDescriptor inside the formatter will have
            # the wrong storage class so we need to create a new one given
            # the immutability
            writeStorageClass = rwInfo.info.storageClass

            # We may need to put some thought into parameters for read
            # components but for now forward them on as is
            readFormatter = type(rwInfo.formatter)(FileDescriptor(rwInfo.location,
                                                                  readStorageClass=refStorageClass,
                                                                  storageClass=writeStorageClass,
                                                                  parameters=parameters),
                                                   ref.dataId)

            # The assembler can not receive any parameter requests for a
            # read-only component at this time since the assembler will
            # see the storage class of the read-only component and those
            # parameters will have to be handled by the formatter on the
            # forwarded storage class.
            assemblerParams = {}

            # Need to created a new info that specifies the read-only
            # component and associated storage class
            readInfo = DatastoreFileGetInformation(rwInfo.location, readFormatter,
                                                   rwInfo.info, assemblerParams,
                                                   refComponent, refStorageClass)

            return self._read_artifact_into_memory(readInfo, ref, isComponent=True)

        else:

            # Indicate whether we are reading a component from this
            # artifact using the formatter
            isReadingComponent = False

            if refComponent and refComponent in allComponents:
                # Single file request of a component that exists
                getInfo = allComponents[refComponent]

                # We are only reading a component if this is a composite
                # Else it really is the component.
                isReadingComponent = not isDisassembled

            elif None in allComponents:
                # Asking for the composite
                getInfo = allComponents[None]

                isReadingComponent = refComponent is not None

            else:
                # Should not be possible since we should always store
                # the composite information
                raise FileNotFoundError(f"Component {refComponent} not found "
                                        f"for ref {ref} in datastore {self.name}")

            return self._read_artifact_into_memory(getInfo, ref, isComponent=isReadingComponent)

    @transactional
    def put(self, inMemoryDataset: Any, ref: DatasetRef) -> None:
        """Write a InMemoryDataset with a given `DatasetRef` to the store.

        Parameters
        ----------
        inMemoryDataset : `object`
            The dataset to store.
        ref : `DatasetRef`
            Reference to the associated Dataset.

        Raises
        ------
        TypeError
            Supplied object and storage class are inconsistent.
        DatasetTypeNotSupportedError
            The associated `DatasetType` is not handled by this datastore.

        Notes
        -----
        If the datastore is configured to reject certain dataset types it
        is possible that the put will fail and raise a
        `DatasetTypeNotSupportedError`.  The main use case for this is to
        allow `ChainedDatastore` to put to multiple datastores without
        requiring that every datastore accepts the dataset.
        """

        doDisassembly = self.composites.shouldBeDisassembled(ref)
        # doDisassembly = True

        artifacts = []
        if doDisassembly:
            components = ref.datasetType.storageClass.assembler().disassemble(inMemoryDataset)
            for component, componentInfo in components.items():
                # Don't recurse because we want to take advantage of
                # bulk insert -- need a new DatasetRef that refers to the
                # same dataset_id but has the component DatasetType
                # DatasetType does not refer to the types of components
                # So we construct one ourselves.
                compRef = ref.makeComponentRef(component)
                storedInfo = self._write_in_memory_to_artifact(componentInfo.component, compRef)
                artifacts.append((compRef, storedInfo))
            # in order to allow us to discover the composite Formatter we
            # also need to write an entry with an undefined filename
            # In some edge cases where something is always disassembled we
            # might not know the formatter. In that case we can warn but
            # allow it and fail later if someone tries to get a read-only
            # component.
            try:
                _, formatter = self._prepare_for_put(inMemoryDataset, ref)
            except DatasetTypeNotSupportedError:
                log.warning("No formatter associated with composite dataset (%s). "
                            "Read-only components will not be supported.", ref)
            else:
                compositeInfo = StoredFileInfo(formatter=formatter, path=None,
                                               storageClass=ref.datasetType.storageClass,
                                               component=None, file_size=0, checksum=None)
                artifacts.append((ref, compositeInfo))
        else:
            # Write the entire thing out
            storedInfo = self._write_in_memory_to_artifact(inMemoryDataset, ref)
            artifacts.append((ref, storedInfo))

        self._register_datasets(artifacts)

    @transactional
    def trash(self, ref: DatasetRef, ignore_errors: bool = True) -> None:
        """Indicate to the datastore that a dataset can be removed.

        Parameters
        ----------
        ref : `DatasetRef`
            Reference to the required Dataset.
        ignore_errors : `bool`
            If `True` return without error even if something went wrong.
            Problems could occur if another process is simultaneously trying
            to delete.

        Raises
        ------
        FileNotFoundError
            Attempt to remove a dataset that does not exist.
        """
        # Get file metadata and internal metadata
        log.debug("Trashing %s in datastore %s", ref, self.name)

        fileLocations = self._get_dataset_locations_info(ref)

        if not fileLocations:
            err_msg = f"Requested dataset to trash ({ref}) is not known to datastore {self.name}"
            if ignore_errors:
                log.warning(err_msg)
                return
            else:
                raise FileNotFoundError(err_msg)

        for location, storedFileInfo in fileLocations:
            if storedFileInfo.path is None:
                # Composite formatter
                continue
            if not self._artifact_exists(location):
                err_msg = f"Dataset is known to datastore {self.name} but " \
                          f"associated artifact ({location.uri}) is missing"
                if ignore_errors:
                    log.warning(err_msg)
                    return
                else:
                    raise FileNotFoundError(err_msg)

        # Mark dataset as trashed
        try:
            self._move_to_trash_in_registry(ref)
        except Exception as e:
            if ignore_errors:
                log.warning(f"Attempted to mark dataset ({ref}) to be trashed in datastore {self.name} "
                            f"but encountered an error: {e}")
                pass
            else:
                raise

    @transactional
    def emptyTrash(self, ignore_errors: bool = True) -> None:
        """Remove all datasets from the trash.

        Parameters
        ----------
        ignore_errors : `bool`
            If `True` return without error even if something went wrong.
            Problems could occur if another process is simultaneously trying
            to delete.
        """
        log.debug("Emptying trash in datastore %s", self.name)
        # Context manager will empty trash iff we finish it without raising.
        with self._bridge.emptyTrash() as trashed:
            for ref in trashed:
                fileLocations = self._get_dataset_locations_info(ref)

                if not fileLocations:
                    err_msg = f"Requested dataset ({ref}) does not exist in datastore {self.name}"
                    if ignore_errors:
                        log.warning(err_msg)
                        continue
                    else:
                        raise FileNotFoundError(err_msg)

                for location, storedFileInfo in fileLocations:
                    if storedFileInfo.path is None:
                        # Composite formatter
                        continue

                    if not self._artifact_exists(location):
                        err_msg = f"Dataset {location.uri} no longer present in datastore {self.name}"
                        if ignore_errors:
                            log.warning(err_msg)
                            continue
                        else:
                            raise FileNotFoundError(err_msg)

                    # Can only delete the artifact if there are no references
                    # to the file from untrashed dataset refs.
                    if self._can_remove_dataset_artifact(ref, location):
                        # Point of no return for this artifact
                        log.debug("Removing artifact %s from datastore %s", location.uri, self.name)
                        try:
                            self._delete_artifact(location)
                        except Exception as e:
                            if ignore_errors:
                                log.critical("Encountered error removing artifact %s from datastore %s: %s",
                                             location.uri, self.name, e)
                            else:
                                raise

                # Now must remove the entry from the internal registry even if
                # the artifact removal failed and was ignored,
                # otherwise the removal check above will never be true
                try:
                    # There may be multiple rows associated with this ref
                    # depending on disassembly
                    self.removeStoredItemInfo(ref)
                except Exception as e:
                    if ignore_errors:
                        log.warning("Error removing dataset %s (%s) from internal registry of %s: %s",
                                    ref.id, location.uri, self.name, e)
                        continue
                    else:
                        raise FileNotFoundError(err_msg)

    def validateConfiguration(self, entities: Iterable[Union[DatasetRef, DatasetType, StorageClass]],
                              logFailures: bool = False) -> None:
        """Validate some of the configuration for this datastore.

        Parameters
        ----------
        entities : iterable of `DatasetRef`, `DatasetType`, or `StorageClass`
            Entities to test against this configuration.  Can be differing
            types.
        logFailures : `bool`, optional
            If `True`, output a log message for every validation error
            detected.

        Raises
        ------
        DatastoreValidationError
            Raised if there is a validation problem with a configuration.
            All the problems are reported in a single exception.

        Notes
        -----
        This method checks that all the supplied entities have valid file
        templates and also have formatters defined.
        """

        templateFailed = None
        try:
            self.templates.validateTemplates(entities, logFailures=logFailures)
        except FileTemplateValidationError as e:
            templateFailed = str(e)

        formatterFailed = []
        for entity in entities:
            try:
                self.formatterFactory.getFormatterClass(entity)
            except KeyError as e:
                formatterFailed.append(str(e))
                if logFailures:
                    log.fatal("Formatter failure: %s", e)

        if templateFailed or formatterFailed:
            messages = []
            if templateFailed:
                messages.append(templateFailed)
            if formatterFailed:
                messages.append(",".join(formatterFailed))
            msg = ";\n".join(messages)
            raise DatastoreValidationError(msg)

    def getLookupKeys(self) -> Set[LookupKey]:
        # Docstring is inherited from base class
        return self.templates.getLookupKeys() | self.formatterFactory.getLookupKeys() | \
            self.constraints.getLookupKeys()

    def validateKey(self, lookupKey: LookupKey,
                    entity: Union[DatasetRef, DatasetType, StorageClass]) -> None:
        # Docstring is inherited from base class
        # The key can be valid in either formatters or templates so we can
        # only check the template if it exists
        if lookupKey in self.templates:
            try:
                self.templates[lookupKey].validateTemplate(entity)
            except FileTemplateValidationError as e:
                raise DatastoreValidationError(e) from e
