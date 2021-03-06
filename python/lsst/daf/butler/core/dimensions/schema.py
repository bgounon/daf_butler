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
    "addDimensionForeignKey",
    "REGION_FIELD_SPEC",
)

import copy

from typing import Tuple, Type, TYPE_CHECKING

from .. import ddl
from ..named import NamedValueSet
from ..timespan import DatabaseTimespanRepresentation

if TYPE_CHECKING:  # Imports needed only for type annotations; may be circular.
    from .elements import DimensionElement, Dimension


# Most regions are small (they're quadrilaterals), but visit ones can be quite
# large because they have a complicated boundary.  For HSC, about ~1400 bytes.
REGION_FIELD_SPEC = ddl.FieldSpec(name="region", nbytes=2048, dtype=ddl.Base64Region)


def _makeForeignKeySpec(dimension: Dimension) -> ddl.ForeignKeySpec:
    """Make a `ddl.ForeignKeySpec` that references the table for the given
    `Dimension` table.

    Most callers should use the higher-level `addDimensionForeignKey` function
    instead.

    Parameters
    ----------
    dimension : `Dimension`
        The dimension to be referenced.  Caller guarantees that it is actually
        associated with a table.

    Returns
    -------
    spec : `ddl.ForeignKeySpec`
        A database-agnostic foreign key specification.
    """
    source = []
    target = []
    for other in dimension.required:
        if other == dimension:
            target.append(dimension.primaryKey.name)
        else:
            target.append(other.name)
        source.append(other.name)
    return ddl.ForeignKeySpec(table=dimension.name, source=tuple(source), target=tuple(target))


def addDimensionForeignKey(tableSpec: ddl.TableSpec, dimension: Dimension, *,
                           primaryKey: bool, nullable: bool = False, constraint: bool = True
                           ) -> ddl.FieldSpec:
    """Add a field and possibly a foreign key to a table specification that
    reference the table for the given `Dimension`.

    Parameters
    ----------
    tableSpec : `ddl.TableSpec`
        Specification the field and foreign key are to be added to.
    dimension : `Dimension`
        Dimension to be referenced.  If this dimension has required
        dependencies, those must have already been added to the table.  A field
        will be added that correspond to this dimension's primary key, and a
        foreign key constraint will be added only if the dimension is
        associated with a table of its own.
    primaryKey : `bool`
        If `True`, the new field will be added as part of a compound primary
        key for the table.
    nullable : `bool`, optional
        If `False` (default) the new field will be added with a NOT NULL
        constraint.
    constraint : `bool`
        If `False` (`True` is default), just add the field, not the foreign
        key constraint.

    Returns
    -------
    fieldSpec : `ddl.FieldSpec`
        Specification for the field just added.
    """
    # Add the dependency's primary key field, but use the dimension name for
    # the field name to make it unique and more meaningful in this table.
    fieldSpec = copy.copy(dimension.primaryKey)
    fieldSpec.name = dimension.name
    fieldSpec.primaryKey = primaryKey
    fieldSpec.nullable = nullable
    tableSpec.fields.add(fieldSpec)
    # Also add a foreign key constraint on the dependency table, but only if
    # there actually is one and we weren't told not to.
    if dimension.hasTable() and dimension.viewOf is None and constraint:
        tableSpec.foreignKeys.append(_makeForeignKeySpec(dimension))
    return fieldSpec


class DimensionElementFields:
    """An object that constructs the table schema for a `DimensionElement` and
    provides a categorized view of its fields.

    Parameters
    ----------
    element : `DimensionElement`
        Element for which to make a table specification.

    Notes
    -----
    This combines the foreign key fields from dependencies, unique keys
    for true `Dimension` instances, metadata fields, and region/timestamp
    fields for spatial/temporal elements.

    Callers should use `DimensionUniverse.makeSchemaSpec` if they want to
    account for elements that have no table or reference another table; this
    class simply creates a specification for the table an element _would_ have
    without checking whether it does have one.  That can be useful in contexts
    (e.g. `DimensionRecord`) where we want to simulate the existence of such a
    table.
    """
    def __init__(self, element: DimensionElement):
        self.element = element
        self._tableSpec = ddl.TableSpec(fields=())
        # Add the primary key fields of required dimensions.  These continue to
        # be primary keys in the table for this dimension.
        self.required = NamedValueSet()
        self.dimensions = NamedValueSet()
        self.standard = NamedValueSet()
        dependencies = []
        for dimension in element.required:
            if dimension != element:
                fieldSpec = addDimensionForeignKey(self._tableSpec, dimension, primaryKey=True)
                dependencies.append(fieldSpec.name)
            else:
                fieldSpec = element.primaryKey  # type: ignore
                # A Dimension instance is in its own required dependency graph
                # (always at the end, because of topological ordering).  In
                # this case we don't want to rename the field.
                self._tableSpec.fields.add(fieldSpec)
            self.required.add(fieldSpec)
            self.dimensions.add(fieldSpec)
            self.standard.add(fieldSpec)
        # Add fields and foreign keys for implied dimensions.  These are
        # primary keys in their own table, but should not be here.  As with
        # required dependencies, we rename the fields with the dimension name.
        # We use element.implied instead of element.graph.implied because we
        # don't want *recursive* implied dependencies.
        self.implied = NamedValueSet()
        for dimension in element.implied:
            fieldSpec = addDimensionForeignKey(self._tableSpec, dimension, primaryKey=False, nullable=True)
            self.implied.add(fieldSpec)
            self.dimensions.add(fieldSpec)
            self.standard.add(fieldSpec)
        # Add non-primary unique keys and unique constraints for them.
        for fieldSpec in getattr(element, "alternateKeys", ()):
            self._tableSpec.fields.add(fieldSpec)
            self._tableSpec.unique.add(tuple(dependencies) + (fieldSpec.name,))
            self.standard.add(fieldSpec)
        # Add other metadata fields.
        for fieldSpec in element.metadata:
            self._tableSpec.fields.add(fieldSpec)
            self.standard.add(fieldSpec)
        names = list(self.standard.names)
        # Add fields for regions and/or timespans.
        if element.spatial is not None:
            self._tableSpec.fields.add(REGION_FIELD_SPEC)
            names.append(REGION_FIELD_SPEC.name)
        if element.temporal is not None:
            names.append(DatabaseTimespanRepresentation.NAME)
        self.names = tuple(names)

    def makeTableSpec(self, tsRepr: Type[DatabaseTimespanRepresentation]) -> ddl.TableSpec:
        """Construct a complete specification for a table that could hold the
        records of this element.

        Parameters
        ----------
        tsRepr : `type` (`DatabaseTimespanRepresentation` subclass)
            Class object that specifies how timespans are represented in the
            database.

        Returns
        -------
        spec : `ddl.TableSpec`
            Specification for a table.
        """
        if self.element.temporal is not None:
            spec = ddl.TableSpec(
                fields=NamedValueSet(self._tableSpec.fields),
                unique=self._tableSpec.unique,
                indexes=self._tableSpec.indexes,
                foreignKeys=self._tableSpec.foreignKeys,
            )
            for fieldSpec in tsRepr.makeFieldSpecs(nullable=True):
                spec.fields.add(fieldSpec)
        else:
            spec = self._tableSpec
        return spec

    element: DimensionElement
    """The dimension element these fields correspond to (`DimensionElement`).
    """

    required: NamedValueSet[ddl.FieldSpec]
    """The fields of this table that correspond to the element's required
    dimensions, in that order, i.e. `DimensionElement.required`
    (`NamedValueSet` [ `ddl.FieldSpec` ]).
    """

    implied: NamedValueSet[ddl.FieldSpec]
    """The fields of this table that correspond to the element's implied
    dimensions, in that order, i.e. `DimensionElement.implied`
    (`NamedValueSet` [ `ddl.FieldSpec` ]).
    """

    dimensions: NamedValueSet[ddl.FieldSpec]
    """The fields of this table that correspond to the element's direct
    required and implied dimensions, in that order, i.e.
    `DimensionElement.dimensions` (`NamedValueSet` [ `ddl.FieldSpec` ]).
    """

    standard: NamedValueSet[ddl.FieldSpec]
    """All standard fields that are expected to have the same form in all
    databases; this is all fields other than those that represent a region
    and/or timespan (`NamedValueSet` [ `ddl.FieldSpec` ]).
    """

    names: Tuple[str, ...]
    """The names of all fields in the specification (`tuple` [ `str` ]).

    This includes "region" and/or "timespan" if `element` is spatial and/or
    temporal (respectively).  The actual database representation of these
    quantities may involve multiple fields (or even fields only on a different
    table), but the Python representation of those rows (i.e. `DimensionRecord`
    instances) will always contain exactly these fields.
    """
