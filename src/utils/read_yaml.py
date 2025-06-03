import io
import re
from pathlib import Path
from typing import Any, Union

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    """
    Read yaml file.

    Args:
        path: path to yaml file

    Returns:
        dictionary object with parameters from yaml
    """
    with path.open() as file:
        loaded_yaml = yaml_loader(file)
    return loaded_yaml


def join(loader: yaml.Loader, node: yaml.Node) -> str:
    """
    Joins path given as a list

    Args:
        loader: class that inherit from yaml.Loader class
        node: yaml node that contains a list of path\'s parts

    Returns:
        Joined paths part in form of list
    """
    seq = loader.construct_sequence(node)
    return "/".join(seq)


def yaml_loader(file: Union[str, io.TextIOWrapper]) -> dict[str, Any]:
    """
    Read yaml file. Main change in function allows using 1e-5 notation instead of 0.00001.
    Loader overrides existing one in resolver.py script from pyyaml library. One line for
    scientific notation shown above was added according to
    https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number

    Args:
        file: yaml bytes or string that should be loaded

    Returns:
        dictionary object with parameters from yaml
    """
    # The SafeLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    # custom regex was used to convert '1e-15' from string to number in python
    loader = yaml.SafeLoader
    loader.add_constructor("!join", join)
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    yaml_dict = yaml.load(file, Loader=loader)

    return yaml_dict
