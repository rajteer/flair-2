import io
import re
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    """Read yaml file.

    Args:
        path: path to yaml file

    Returns:
        dictionary object with parameters from yaml

    """
    with path.open() as file:
        return yaml_loader(file)


def join(loader: yaml.Loader, node: yaml.Node) -> str:
    """Join path parts given as a list.

    Args:
        loader: class that inherit from yaml.Loader class
        node: yaml node that contains a list of path's parts

    Returns:
        Joined paths part in form of list

    """
    seq = loader.construct_sequence(node)
    return "/".join(seq)


def yaml_loader(file: str | io.TextIOWrapper) -> dict[str, Any]:
    """Read YAML file.

    Main change in this loader allows using scientific notation (e.g. 1e-5)
    without it being interpreted as a string. This function registers a
    custom constructor and an implicit resolver on pyyaml's SafeLoader.

    Args:
        file: yaml bytes or string that should be loaded

    Returns:
        dictionary object with parameters from yaml

    """
    yaml.SafeLoader.add_constructor("!join", join)
    yaml.SafeLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            r"^(?:"
            r"[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?"
            r"|[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+"
            r"|\.[0-9_]+(?:[eE][-+]?[0-9]+)?"
            r"|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+.\.[0-9_]*"
            r"|[-+]?\.(?:inf|Inf|INF)"
            r"|\.(?:nan|NaN|NAN))$",
            re.VERBOSE,
        ),
        list("-+0123456789."),
    )

    return yaml.safe_load(file)
