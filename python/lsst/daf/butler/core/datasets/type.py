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

__all__ = ["DatasetType"]

from copy import deepcopy
import re

from types import MappingProxyType

from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)


from ..storageClass import StorageClass, StorageClassFactory
from ..dimensions import DimensionGraph
from ..configSupport import LookupKey

if TYPE_CHECKING:
    from ..dimensions import Dimension, DimensionUniverse


def _safeMakeMappingProxyType(data: Optional[Mapping]) -> Mapping:
    if data is None:
        data = {}
    return MappingProxyType(data)


class DatasetType:
    r"""A named category of Datasets that defines how they are organized,
    related, and stored.

    A concrete, final class whose instances represent `DatasetType`\ s.
    `DatasetType` instances may be constructed without a `Registry`,
    but they must be registered
    via `Registry.registerDatasetType()` before corresponding Datasets
    may be added.
    `DatasetType` instances are immutable.

    Parameters
    ----------
    name : `str`
        A string name for the Dataset; must correspond to the same
        `DatasetType` across all Registries.  Names must start with an
        upper or lowercase letter, and may contain only letters, numbers,
        and underscores.  Component dataset types should contain a single
        period separating the base dataset type name from the component name
        (and may be recursive).
    dimensions : `DimensionGraph` or iterable of `Dimension`
        Dimensions used to label and relate instances of this `DatasetType`.
        If not a `DimensionGraph`, ``universe`` must be provided as well.
    storageClass : `StorageClass` or `str`
        Instance of a `StorageClass` or name of `StorageClass` that defines
        how this `DatasetType` is persisted.
    parentStorageClass : `StorageClass` or `str`, optional
        Instance of a `StorageClass` or name of `StorageClass` that defines
        how the composite parent is persisted.  Must be `None` if this
        is not a component. Mandatory if it is a component but can be the
        special temporary placeholder
        (`DatasetType.PlaceholderParentStorageClass`) to allow
        construction with an intent to finalize later.
    universe : `DimensionUniverse`, optional
        Set of all known dimensions, used to normalize ``dimensions`` if it
        is not already a `DimensionGraph`.
    """

    __slots__ = ("_name", "_dimensions", "_storageClass", "_storageClassName",
                 "_parentStorageClass", "_parentStorageClassName")

    VALID_NAME_REGEX = re.compile("^[a-zA-Z][a-zA-Z0-9_]*(\\.[a-zA-Z][a-zA-Z0-9_]*)*$")

    PlaceholderParentStorageClass = StorageClass("PlaceHolder")
    """Placeholder StorageClass that can be used temporarily for a
    component.

    This can be useful in pipeline construction where we are creating
    dataset types without a registry.
    """

    @staticmethod
    def nameWithComponent(datasetTypeName: str, componentName: str) -> str:
        """Form a valid DatasetTypeName from a parent and component.

        No validation is performed.

        Parameters
        ----------
        datasetTypeName : `str`
            Base type name.
        componentName : `str`
            Name of component.

        Returns
        -------
        compTypeName : `str`
            Name to use for component DatasetType.
        """
        return "{}.{}".format(datasetTypeName, componentName)

    def __init__(self, name: str, dimensions: Union[DimensionGraph, Iterable[Dimension]],
                 storageClass: Union[StorageClass, str],
                 parentStorageClass: Optional[Union[StorageClass, str]] = None, *,
                 universe: Optional[DimensionUniverse] = None):
        if self.VALID_NAME_REGEX.match(name) is None:
            raise ValueError(f"DatasetType name '{name}' is invalid.")
        self._name = name
        if not isinstance(dimensions, DimensionGraph):
            if universe is None:
                raise ValueError("If dimensions is not a normalized DimensionGraph, "
                                 "a universe must be provided.")
            dimensions = universe.extract(dimensions)
        self._dimensions = dimensions
        if not isinstance(storageClass, (StorageClass, str)):
            raise ValueError("StorageClass argument must be StorageClass or str. "
                             f"Got {storageClass}")
        self._storageClass: Optional[StorageClass]
        if isinstance(storageClass, StorageClass):
            self._storageClass = storageClass
            self._storageClassName = storageClass.name
        else:
            self._storageClass = None
            self._storageClassName = storageClass

        self._parentStorageClass: Optional[StorageClass] = None
        self._parentStorageClassName: Optional[str] = None
        if parentStorageClass is not None:
            if not isinstance(storageClass, (StorageClass, str)):
                raise ValueError("Parent StorageClass argument must be StorageClass or str. "
                                 f"Got {parentStorageClass}")

            # Only allowed for a component dataset type
            _, componentName = self.splitDatasetTypeName(self._name)
            if componentName is None:
                raise ValueError("Can not specify a parent storage class if this is not a component"
                                 f" ({self._name})")
            if isinstance(parentStorageClass, StorageClass):
                self._parentStorageClass = parentStorageClass
                self._parentStorageClassName = parentStorageClass.name
            else:
                self._parentStorageClassName = parentStorageClass

        # Ensure that parent storage class is specified when we have
        # a component and is not specified when we don't
        _, componentName = self.splitDatasetTypeName(self._name)
        if parentStorageClass is None and componentName is not None:
            raise ValueError(f"Component dataset type '{self._name}' constructed without parent"
                             " storage class")
        if parentStorageClass is not None and componentName is None:
            raise ValueError(f"Parent storage class specified by {self._name} is not a composite")

    def __repr__(self) -> str:
        parent = ""
        if self._parentStorageClassName:
            parent = f", parentStorageClass={self._parentStorageClassName}"
        return f"DatasetType({self.name}, {self.dimensions}, {self._storageClassName}{parent})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self._name != other._name:
            return False
        if self._dimensions != other._dimensions:
            return False
        if self._storageClass is not None and other._storageClass is not None:
            if self._storageClass != other._storageClass:
                return False
        else:
            if self._storageClassName != other._storageClassName:
                return False
        if self._parentStorageClass is not None and other._parentStorageClass is not None:
            return self._parentStorageClass == other._parentStorageClass
        else:
            return self._parentStorageClassName == other._parentStorageClassName

    def __hash__(self) -> int:
        """Hash DatasetType instance.

        This only uses StorageClass name which is it consistent with the
        implementation of StorageClass hash method.
        """
        return hash((self._name, self._dimensions, self._storageClassName,
                     self._parentStorageClassName))

    @property
    def name(self) -> str:
        """A string name for the Dataset; must correspond to the same
        `DatasetType` across all Registries.
        """
        return self._name

    @property
    def dimensions(self) -> DimensionGraph:
        r"""The `Dimension`\ s that label and relate instances of this
        `DatasetType` (`DimensionGraph`).
        """
        return self._dimensions

    @property
    def storageClass(self) -> StorageClass:
        """`StorageClass` instance that defines how this `DatasetType`
        is persisted. Note that if DatasetType was constructed with a name
        of a StorageClass then Butler has to be initialized before using
        this property.
        """
        if self._storageClass is None:
            self._storageClass = StorageClassFactory().getStorageClass(self._storageClassName)
        return self._storageClass

    @property
    def parentStorageClass(self) -> Optional[StorageClass]:
        """`StorageClass` instance that defines how the composite associated
        with this  `DatasetType` is persisted.

        Note that if DatasetType was constructed with a name of a
        StorageClass then Butler has to be initialized before using this
        property. Can be `None` if this is not a component of a composite.
        Must be defined if this is a component.
        """
        if self._parentStorageClass is None and self._parentStorageClassName is None:
            return None
        if self._parentStorageClass is None and self._parentStorageClassName is not None:
            self._parentStorageClass = StorageClassFactory().getStorageClass(self._parentStorageClassName)
        return self._parentStorageClass

    def finalizeParentStorageClass(self, newParent: StorageClass) -> None:
        """Replace the current placeholder parent storage class with
        the real parent.

        Parameters
        ----------
        newParent : `StorageClass`
            The new parent to be associated with this composite dataset
            type.  This replaces the temporary placeholder parent that
            was specified during construction.

        Raises
        ------
        ValueError
            Raised if this dataset type is not a component of a composite.
            Raised if a StorageClass is not given.
            Raised if the parent currently associated with the dataset
            type is not a placeholder.
        """
        if not self.isComponent():
            raise ValueError("Can not set a parent storage class if this is not a component"
                             f" ({self.name})")
        if self._parentStorageClass != self.PlaceholderParentStorageClass:
            raise ValueError(f"This DatasetType has a parent of {self._parentStorageClassName} and"
                             " is not a placeholder.")
        if not isinstance(newParent, StorageClass):
            raise ValueError(f"Supplied parent must be a StorageClass. Got {newParent!r}")
        self._parentStorageClass = newParent
        self._parentStorageClassName = newParent.name

    @staticmethod
    def splitDatasetTypeName(datasetTypeName: str) -> Tuple[str, Optional[str]]:
        """Given a dataset type name, return the root name and the component
        name.

        Parameters
        ----------
        datasetTypeName : `str`
            The name of the dataset type, can include a component using
            a "."-separator.

        Returns
        -------
        rootName : `str`
            Root name without any components.
        componentName : `str`
            The component if it has been specified, else `None`.

        Notes
        -----
        If the dataset type name is ``a.b.c`` this method will return a
        root name of ``a`` and a component name of ``b.c``.
        """
        comp = None
        root = datasetTypeName
        if "." in root:
            # If there is doubt, the component is after the first "."
            root, comp = root.split(".", maxsplit=1)
        return root, comp

    def nameAndComponent(self) -> Tuple[str, Optional[str]]:
        """Return the root name of this dataset type and the component
        name (if defined).

        Returns
        -------
        rootName : `str`
            Root name for this `DatasetType` without any components.
        componentName : `str`
            The component if it has been specified, else `None`.
        """
        return self.splitDatasetTypeName(self.name)

    def component(self) -> Optional[str]:
        """Component name (if defined)

        Returns
        -------
        comp : `str`
            Name of component part of DatasetType name. `None` if this
            `DatasetType` is not associated with a component.
        """
        _, comp = self.nameAndComponent()
        return comp

    def componentTypeName(self, component: str) -> str:
        """Given a component name, derive the datasetTypeName of that component

        Parameters
        ----------
        component : `str`
            Name of component

        Returns
        -------
        derived : `str`
            Compound name of this `DatasetType` and the component.

        Raises
        ------
        KeyError
            Requested component is not supported by this `DatasetType`.
        """
        if component in self.storageClass.allComponents():
            return self.nameWithComponent(self.name, component)
        raise KeyError("Requested component ({}) not understood by this DatasetType".format(component))

    def makeComponentDatasetType(self, component: str) -> DatasetType:
        """Return a DatasetType suitable for the given component, assuming the
        same dimensions as the parent.

        Parameters
        ----------
        component : `str`
            Name of component

        Returns
        -------
        datasetType : `DatasetType`
            A new DatasetType instance.
        """
        # The component could be a read/write or read component
        return DatasetType(self.componentTypeName(component), dimensions=self.dimensions,
                           storageClass=self.storageClass.allComponents()[component],
                           parentStorageClass=self.storageClass)

    def makeAllComponentDatasetTypes(self) -> List[DatasetType]:
        """Return all the component dataset types assocaited with this
        dataset type.

        Returns
        -------
        all : `list` of `DatasetType`
            All the component dataset types. If this is not a composite
            then returns an empty list.
        """
        return [self.makeComponentDatasetType(componentName)
                for componentName in self.storageClass.allComponents()]

    def isComponent(self) -> bool:
        """Boolean indicating whether this `DatasetType` refers to a
        component of a composite.

        Returns
        -------
        isComponent : `bool`
            `True` if this `DatasetType` is a component, `False` otherwise.
        """
        if self.component():
            return True
        return False

    def isComposite(self) -> bool:
        """Boolean indicating whether this `DatasetType` is a composite type.

        Returns
        -------
        isComposite : `bool`
            `True` if this `DatasetType` is a composite type, `False`
            otherwise.
        """
        return self.storageClass.isComposite()

    def _lookupNames(self) -> Tuple[LookupKey, ...]:
        """Name keys to use when looking up this datasetType in a
        configuration.

        The names are returned in order of priority.

        Returns
        -------
        names : `tuple` of `LookupKey`
            Tuple of the `DatasetType` name and the `StorageClass` name.
            If the name includes a component the name with the component
            is first, then the name without the component and finally
            the storage class name.
        """
        rootName, componentName = self.nameAndComponent()
        lookups: Tuple[LookupKey, ...] = (LookupKey(name=self.name),)
        if componentName is not None:
            lookups = lookups + (LookupKey(name=rootName),)

        if self.dimensions:
            # Dimensions are a lower priority than dataset type name
            lookups = lookups + (LookupKey(dimensions=self.dimensions),)

        return lookups + self.storageClass._lookupNames()

    def __reduce__(self) -> Tuple[Type[DatasetType], Tuple[str, DimensionGraph, str, Optional[str]]]:
        """Support pickling.

        StorageClass instances can not normally be pickled, so we pickle
        StorageClass name instead of instance.
        """
        return (DatasetType, (self.name, self.dimensions, self._storageClassName,
                              self._parentStorageClassName))

    def __deepcopy__(self, memo: Any) -> DatasetType:
        """Support for deep copy method.

        Normally ``deepcopy`` will use pickle mechanism to make copies.
        We want to avoid that to support (possibly degenerate) use case when
        DatasetType is constructed with StorageClass instance which is not
        registered with StorageClassFactory (this happens in unit tests).
        Instead we re-implement ``__deepcopy__`` method.
        """
        return DatasetType(name=deepcopy(self.name, memo),
                           dimensions=deepcopy(self.dimensions, memo),
                           storageClass=deepcopy(self._storageClass or self._storageClassName, memo),
                           parentStorageClass=deepcopy(self._parentStorageClass
                                                       or self._parentStorageClassName, memo))
