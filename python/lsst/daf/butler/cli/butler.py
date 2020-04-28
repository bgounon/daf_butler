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

import click
from collections import defaultdict
import logging
import os
import yaml

from . import cmd as butlerCommands
from .utils import to_upper
from lsst.utils import doImport

localCmdPkg = "lsst.daf.butler.cli.cmd"

log = logging.getLogger(__name__)


def _initLogging(logLevel):
    numeric_level = getattr(logging, logLevel, None)
    if not isinstance(numeric_level, int):
        raise click.ClickException(f"Invalid log level: {logLevel}")
    logging.basicConfig(level=numeric_level)


def funcNameToCmdName(functionName):
    """Change underscores, used in functions, to dashes, used in commands."""
    return functionName.replace("_", "-")


def cmdNameToFuncName(commandName):
    """Change dashes, used in commands, to underscores, used in functions."""
    return commandName.replace("-", "_")


class LoaderCLI(click.MultiCommand):

    def __init__(self, *args, **kwargs):
        self.commands = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def _getPluginList():
        """Get the list of importable yaml files that contain butler cli data.

        Returns
        -------
        `list` [`str`]
            The list of files that contain yaml data about a cli plugin.
        """
        pluginModules = os.environ.get("DAF_BUTLER_PLUGINS")
        if pluginModules:
            return pluginModules.split(":")
        return []

    @staticmethod
    def _importPlugin(pluginName):
        """Import a plugin that contains Click commands.

        Parameters
        ----------
        pluginName : string
            An importable module whose __all__ parameter contains the commands
            that can be called.

        Returns
        -------
        An imported module or None
            The imported module, or None if the module could not be imported.
        """
        try:
            return doImport(pluginName)
        except (TypeError, ModuleNotFoundError, ImportError) as err:
            log.warning("Could not import plugin from %s, skipping.", pluginName)
            log.debug("Plugin import exception: %s", err)
            return None

    @staticmethod
    def _mergeCommandLists(a, b):
        """Combine two dicts whose keys are strings (command name) and values
        are list of string (the package(s) that provide the named command).

        Parameters
        ----------
        a : `defaultdict` [`str`: `list` [`str`]]
            The key is the command name. The value is a list of package(s) that
            contains the command.
        b : (same as a)

        Returns
        -------
        commands : `defaultdict` [`str`: [`str`]]
            For convenience, returns a extended with b. ('a' is modified in
            place.)
        """
        for key, val in b.items():
            a[key].extend(val)
        return a

    @staticmethod
    def _getLocalCommands():
        """Get the commands offered by daf_butler.

        Returns
        -------
        commands : `defaultdict` [`str`, `list` [`str`]]
            The key is the command name. The value is a list of package(s) that
            contains the command.
        """
        return defaultdict(list, {funcNameToCmdName(f): [localCmdPkg] for f in butlerCommands.__all__})

    @classmethod
    def _getPluginCommands(cls):
        """Get the commands offered by plugin packages.

        Returns
        -------
        commands : `defaultdict` [`str`, `list` [`str`]]
            The key is the command name. The value is a list of package(s) that
            contains the command.
        """
        commands = defaultdict(list)
        for pluginName in cls._getPluginList():
            try:
                with open(pluginName, "r") as resourceFile:
                    resources = defaultdict(list, yaml.safe_load(resourceFile))
            except Exception as err:
                log.warning(f"Error loading commands from {pluginName}, skipping. {err}")
                continue
            if 'cmd' not in resources:
                log.warning(f"No commands found in {pluginName}, skipping.")
                continue
            pluginCommands = {cmd: [resources["cmd"]["import"]] for cmd in resources["cmd"]["commands"]}
            cls._mergeCommandLists(commands, defaultdict(list, pluginCommands))
        return commands

    @classmethod
    def _getCommands(cls):
        """Get the commands offered by daf_butler and plugin packages.

        Returns
        -------
        commands : `defaultdict` [`str`, `list` [`str`]]
            The key is the command name. The value is a list of package(s) that
            contains the command.
        """
        commands = cls._mergeCommandLists(cls._getLocalCommands(), cls._getPluginCommands())
        return commands

    @staticmethod
    def _raiseIfDuplicateCommands(commands):
        """If any provided command is offered by more than one package raise an
        exception.

        Parameters
        ----------
        commands : `defaultdict` [`str`, `list` [`str`]]
            The key is the command name. The value is a list of package(s) that
            contains the command.

        Raises
        ------
        click.ClickException
            Raised if a command is offered by more than one package, with an
            error message to be displayed to the user.
        """

        msg = ""
        for command, packages in commands.items():
            if len(packages) > 1:
                msg += f"Command '{command}' exists in packages {', '.join(packages)}. "
        if msg:
            raise click.ClickException(msg + "Duplicate commands are not supported, aborting.")

    def list_commands(self, ctx):
        """Used by Click to get all the commands that can be called by the
        butler command, it is used to generate the --help output.

        Parameters
        ----------
        ctx : click.Context
            The current Click context.

        Returns
        -------
        commands : `list` [`str`]
            The names of the commands that can be called by the butler command.
        """
        if self.commands is None:
            self.commands = self._getCommands()
        self._raiseIfDuplicateCommands(self.commands)
        log.debug(self.commands.keys())
        return self.commands.keys()

    def get_command(self, context, name):
        """Used by Click to get a single command for execution.

        Parameters
        ----------
        ctx : click.Context
            The current Click context.
        name : string
            The name of the command to return.

        Returns
        -------
        command : click.Command
            A Command that wraps a callable command function.
        """
        if self.commands is None:
            self.commands = self._getCommands()
        if name not in self.commands:
            return None
        self._raiseIfDuplicateCommands(self.commands)
        if self.commands[name][0] == localCmdPkg:
            return getattr(butlerCommands, cmdNameToFuncName(name))
        return doImport(self.commands[name][0] + "." + cmdNameToFuncName(name))


@click.command(cls=LoaderCLI)
@click.option("--log-level",
              type=click.Choice(["critical", "error", "warning", "info", "debug",
                                 "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]),
              default="warning",
              help="The Python log level to use.",
              callback=to_upper)
def cli(log_level):
    _initLogging(log_level)


def main():
    return cli()
