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

import abc
import click
from collections import defaultdict
import logging
import os
import yaml

from .utils import to_upper
from lsst.utils import doImport


log = logging.getLogger(__name__)


class LoaderCLI(click.MultiCommand, abc.ABC):

    @property
    @abc.abstractmethod
    def localCmdPkg(self):
        """localCmdPkg identifies the location of the commands that are in this
        package. `getLocalCommands` assumes that the commands can be found in
        `localCmdPkg.__all__`, if this is not the case then getLocalCommands
        should be overrideen.

        Returns
        -------
        package : `str`
            The fully qualified location of this package.
        """
        pass

    def getLocalCommands(self):
        """Get the commands offered by the local package. This assumes that the
        commands can be found in `localCmdPkg.__all__`, if this is not the case
        then this function should be overrideen.

        Returns
        -------
        commands : `defaultdict` [`str`, `list` [`str`]]
            The key is the command name. The value is a list of package(s) that
            contains the command.
        """
        commandsLocation = self._importPlugin(self.localCmdPkg)
        if commandsLocation is None:
            # _importPlugins logs an error, don't need to do it again here.
            return {}
        return defaultdict(list, {self._funcNameToCmdName(f):
                                  [self.localCmdPkg] for f in commandsLocation.__all__})

    def list_commands(self, ctx):
        """Used by Click to get all the commands that can be called by the
        butler command, it is used to generate the --help output.

        Parameters
        ----------
        ctx : `click.Context`
            The current Click context.

        Returns
        -------
        commands : `list` [`str`]
            The names of the commands that can be called by the butler command.
        """
        commands = self._getCommands()
        self._raiseIfDuplicateCommands(commands)
        return sorted(commands)

    def get_command(self, context, name):
        """Used by Click to get a single command for execution.

        Parameters
        ----------
        ctx : `click.Context`
            The current Click context.
        name : `str`
            The name of the command to return.

        Returns
        -------
        command : `click.Command`
            A Command that wraps a callable command function.
        """
        commands = self._getCommands()
        if name not in commands:
            return None
        self._raiseIfDuplicateCommands(commands)
        return self._importPlugin(commands[name][0] + "." + self._cmdNameToFuncName(name))

    @classmethod
    def initLogging(cls, logLevel):
        """Initialize the logging system.

        Parameters
        ----------
        logLevel : `str`
            The name of one of the python logging levels.

        Raises
        ------
        click.ClickException
            If the log level can not be processed.
        """
        numeric_level = getattr(logging, logLevel, None)
        if not isinstance(numeric_level, int):
            raise click.ClickException(f"Invalid log level: {logLevel}")
        logging.basicConfig(level=numeric_level)

    @staticmethod
    def getPluginList():
        """Get the list of importable yaml files that contain cli data for this
        command.

        Returns
        -------
        `list` [`str`]
            The list of files that contain yaml data about a cli plugin.
        """
        return []

    @classmethod
    def _funcNameToCmdName(cls, functionName):
        """Convert function name to the butler command name: change
        underscores, (used in functions) to dashes (used in commands), and
        change local-package command names that conflict with python keywords
        to a leagal function name.
        """
        return functionName.replace("_", "-")

    @classmethod
    def _cmdNameToFuncName(cls, commandName):
        """Convert butler command name to function name: change dashes (used in
        commands) to underscores (used in functions), and for local-package
        commands names that conflict with python keywords, change the local,
        legal, function name to the command name."""
        return commandName.replace("-", "_")

    @staticmethod
    def _importPlugin(pluginName):
        """Import a plugin that contains Click commands.

        Parameters
        ----------
        pluginName : `str`
            An importable module whose __all__ parameter contains the commands
            that can be called.

        Returns
        -------
        An imported module or None
            The imported module, or None if the module could not be imported.
        """
        try:
            return doImport(pluginName)
        except Exception as err:
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
        for pluginName in cls.getPluginList():
            try:
                with open(pluginName, "r") as resourceFile:
                    resources = defaultdict(list, yaml.safe_load(resourceFile))
            except Exception as err:
                log.warning(f"Error loading commands from {pluginName}, skipping. {err}")
                continue
            if "cmd" not in resources:
                log.warning(f"No commands found in {pluginName}, skipping.")
                continue
            pluginCommands = {cmd: [resources["cmd"]["import"]] for cmd in resources["cmd"]["commands"]}
            cls._mergeCommandLists(commands, defaultdict(list, pluginCommands))
        return commands

    def _getCommands(self):
        """Get the commands offered by daf_butler and plugin packages.

        Returns
        -------
        commands : `defaultdict` [`str`, `list` [`str`]]
            The key is the command name. The value is a list of package(s) that
            contains the command.
        """
        return self._mergeCommandLists(self.getLocalCommands(), self._getPluginCommands())

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


class ButlerCLI(LoaderCLI):

    localCmdPkg = "lsst.daf.butler.cli.cmd"

    @classmethod
    def _funcNameToCmdName(cls, functionName):
        """Convert function name to the butler command name: change
        underscores, (used in functions) to dashes (used in commands), and
        change local-package command names that conflict with python keywords
        to a leagal function name.
        """
        # The "import" command name and "butler_import" function name are
        # defined in cli/cmd/commands.py, and if those names are changed they
        # must be changed here as well.
        # It is expected that there will be very few butler command names that
        # need to be changed because of e.g. conflicts with python keywords (as
        # is done here and in _cmdNameToFuncName for the 'import' command). If
        # this becomes a common need then some way of doing this should be
        # invented that is better than hard coding the function names into
        # these conversion functions. An extension of the 'cli/resources.yaml'
        # file (as is currently used in obs_base) might be a good way to do it.
        if functionName == "butler_import":
            return "import"
        return super()._funcNameToCmdName(functionName)

    @classmethod
    def _cmdNameToFuncName(cls, commandName):
        """Convert butler command name to function name: change dashes (used in
        commands) to underscores (used in functions), and for local-package
        commands names that conflict with python keywords, change the local,
        legal, function name to the command name."""
        if commandName == "import":
            return "butler_import"
        return super()._cmdNameToFuncName(commandName)

    @staticmethod
    def getPluginList():
        """Get the list of importable yaml files that contain butler cli data.

        Returns
        -------
        `list` [`str`]
            The list of files that contain yaml data about a cli plugin.
        """
        pluginModules = os.environ.get("DAF_BUTLER_PLUGINS")
        if pluginModules:
            return [p for p in pluginModules.split(":") if p != '']
        return []


@click.command(cls=ButlerCLI, context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--log-level",
              type=click.Choice(["critical", "error", "warning", "info", "debug",
                                 "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]),
              default="warning",
              help="The Python log level to use.",
              callback=to_upper)
def cli(log_level):
    ButlerCLI.initLogging(log_level)


def main():
    return cli()
