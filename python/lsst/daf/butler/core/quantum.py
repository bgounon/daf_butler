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

__all__ = ("Quantum",)

from typing import (
    Iterable,
    List,
    Mapping,
    Optional,
    Type,
    TYPE_CHECKING,
    Union,
)

import astropy.time

from lsst.utils import doImport

from .named import NamedKeyDict

if TYPE_CHECKING:
    from .dimensions import DataCoordinate
    from .datasets import DatasetRef, DatasetType


class Quantum:
    """A discrete unit of work that may depend on one or more datasets and
    produces one or more datasets.

    Most Quanta will be executions of a particular ``PipelineTask``’s
    ``runQuantum`` method, but they can also be used to represent discrete
    units of work performed manually by human operators or other software
    agents.

    Parameters
    ----------
    taskName : `str`, optional
        Fully-qualified name of the Task class that executed or will execute
        this Quantum.  If not provided, ``taskClass`` must be.
    taskClass : `type`, optional
        The Task class that executed or will execute this Quantum.  If not
        provided, ``taskName`` must be.  Overrides ``taskName`` if both are
        provided.
    dataId : `DataId`, optional
        The dimension values that identify this `Quantum`.
    run : `str`, optional
        The name of the run this Quantum is a part of.
    initInputs : collection of `DatasetRef`, optional
        Datasets that are needed to construct an instance of the Task.  May
        be a flat iterable of `DatasetRef` instances or a mapping from
        `DatasetType` to `DatasetRef`.
    predictedInputs : `~collections.abc.Mapping`, optional
        Inputs identified prior to execution, organized as a mapping from
        `DatasetType` to a list of `DatasetRef`.  Must be a superset of
        ``actualInputs``.
    actualInputs : `~collections.abc.Mapping`, optional
        Inputs actually used during execution, organized as a mapping from
        `DatasetType` to a list of `DatasetRef`.  Must be a subset of
        ``predictedInputs``.
    outputs : `~collections.abc.Mapping`, optional
        Outputs from executing this quantum of work, organized as a mapping
        from `DatasetType` to a list of `DatasetRef`.
    startTime : `astropy.time.Time`
        The start time for the quantum.
    endTime : `astropy.time.Time`
        The end time for the quantum.
    host : `str`
        The system on this quantum was executed.
    id : `int`, optional
        Unique integer identifier for this quantum.  Usually set to `None`
        (default) and assigned by `Registry`.
    """

    __slots__ = ("_taskName", "_taskClass", "_dataId", "_run",
                 "_initInputs", "_predictedInputs", "_actualInputs", "_outputs",
                 "_id", "_startTime", "_endTime", "_host")

    def __init__(self, *, taskName: Optional[str] = None,
                 taskClass: Optional[Type] = None,
                 dataId: Optional[DataCoordinate] = None,
                 run: Optional[str] = None,
                 initInputs: Optional[Union[Mapping[DatasetType, DatasetRef], Iterable[DatasetRef]]] = None,
                 predictedInputs: Optional[Mapping[DatasetType, List[DatasetRef]]] = None,
                 actualInputs: Optional[Mapping[DatasetType, List[DatasetRef]]] = None,
                 outputs: Optional[Mapping[DatasetType, List[DatasetRef]]] = None,
                 startTime: Optional[astropy.time.Time] = None,
                 endTime: Optional[astropy.time.Time] = None,
                 host: Optional[str] = None,
                 id: Optional[int] = None):
        if taskClass is not None:
            taskName = f"{taskClass.__module__}.{taskClass.__name__}"
        self._taskName = taskName
        self._taskClass = taskClass
        self._run = run
        self._dataId = dataId
        if initInputs is None:
            initInputs = {}
        elif not isinstance(initInputs, Mapping):
            initInputs = {ref.datasetType: ref for ref in initInputs}
        if predictedInputs is None:
            predictedInputs = {}
        if actualInputs is None:
            actualInputs = {}
        if outputs is None:
            outputs = {}
        self._initInputs: NamedKeyDict[DatasetType, DatasetRef] = NamedKeyDict(initInputs)
        self._predictedInputs: NamedKeyDict[DatasetType, List[DatasetRef]] = NamedKeyDict(predictedInputs)
        self._actualInputs: NamedKeyDict[DatasetType, List[DatasetRef]] = NamedKeyDict(actualInputs)
        self._outputs: NamedKeyDict[DatasetType, List[DatasetRef]] = NamedKeyDict(outputs)
        self._id = id
        self._startTime = startTime
        self._endTime = endTime
        self._host = host

    @property
    def taskClass(self) -> Optional[Type]:
        """Task class associated with this `Quantum` (`type`).
        """
        if self._taskClass is None:
            self._taskClass = doImport(self._taskName)
        return self._taskClass

    @property
    def taskName(self) -> Optional[str]:
        """Fully-qualified name of the task associated with `Quantum` (`str`).
        """
        return self._taskName

    @property
    def run(self) -> Optional[str]:
        """The name of the run this Quantum is a part of (`str`).
        """
        return self._run

    @property
    def dataId(self) -> Optional[DataCoordinate]:
        """The dimension values of the unit of processing (`DataId`).
        """
        return self._dataId

    @property
    def initInputs(self) -> NamedKeyDict[DatasetType, DatasetRef]:
        """A mapping of datasets used to construct the Task,
        with `DatasetType` instances as keys (names can also be used for
        lookups) and `DatasetRef` instances as values.
        """
        return self._initInputs

    @property
    def predictedInputs(self) -> NamedKeyDict[DatasetType, List[DatasetRef]]:
        """A mapping of input datasets that were expected to be used,
        with `DatasetType` instances as keys (names can also be used for
        lookups) and a list of `DatasetRef` instances as values.

        Notes
        -----
        We cannot use `set` instead of `list` for the nested container because
        `DatasetRef` instances cannot be compared reliably when some have
        integers IDs and others do not.
        """
        return self._predictedInputs

    @property
    def actualInputs(self) -> NamedKeyDict[DatasetType, List[DatasetRef]]:
        """A mapping of input datasets that were actually used, with the same
        form as `Quantum.predictedInputs`.

        Notes
        -----
        We cannot use `set` instead of `list` for the nested container because
        `DatasetRef` instances cannot be compared reliably when some have
        integers IDs and others do not.
        """
        return self._actualInputs

    @property
    def outputs(self) -> NamedKeyDict[DatasetType, List[DatasetRef]]:
        """A mapping of output datasets (to be) generated for this quantum,
        with the same form as `predictedInputs`.

        Notes
        -----
        We cannot use `set` instead of `list` for the nested container because
        `DatasetRef` instances cannot be compared reliably when some have
        integers IDs and others do not.
        """
        return self._outputs

    def addPredictedInput(self, ref: DatasetRef) -> None:
        """Add an input `DatasetRef` to the `Quantum`.

        This does not automatically update a `Registry`; all `predictedInputs`
        must be present before a `Registry.addQuantum()` is called.

        Parameters
        ----------
        ref : `DatasetRef`
            Reference for a Dataset to add to the Quantum's predicted inputs.
        """
        self._predictedInputs.setdefault(ref.datasetType, []).append(ref)

    def _markInputUsed(self, ref: DatasetRef) -> None:
        """Mark an input as used.

        This does not automatically update a `Registry`.
        For that use `Registry.markInputUsed()` instead.
        """
        # First validate against predicted
        if ref.datasetType not in self._predictedInputs:
            raise ValueError(f"Dataset type {ref.datasetType.name} not in predicted inputs")
        if ref not in self._predictedInputs[ref.datasetType]:
            raise ValueError(f"Actual input {ref} was not predicted")
        # Now insert as actual
        self._actualInputs.setdefault(ref.datasetType, []).append(ref)

    def addOutput(self, ref: DatasetRef) -> None:
        """Add an output `DatasetRef` to the `Quantum`.

        This does not automatically update a `Registry`; all `outputs`
        must be present before a `Registry.addQuantum()` is called.

        Parameters
        ----------
        ref : `DatasetRef`
            Reference for a Dataset to add to the Quantum's outputs.
        """
        self._outputs.setdefault(ref.datasetType, []).append(ref)

    @property
    def id(self) -> Optional[int]:
        """Unique (autoincrement) integer for this quantum (`int`).
        """
        return self._id

    @property
    def startTime(self) -> Optional[astropy.time.Time]:
        """Begin timestamp for the execution of this quantum
        (`astropy.time.Time`).
        """
        return self._startTime

    @property
    def endTime(self) -> Optional[astropy.time.Time]:
        """End timestamp for the execution of this quantum
        (`astropy.time.Time`).
        """
        return self._endTime

    @property
    def host(self) -> Optional[str]:
        """Name of the system on which this quantum was executed (`str`).
        """
        return self._host
