"""General SAX Utilities."""

from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from functools import partial, wraps
from hashlib import md5
from pathlib import Path
from typing import (
    Any,
    NamedTuple,
    TypeVar,
    Unpack,
    cast,
    get_args,
    get_origin,
    overload,
)

import jax
import jax.numpy as jnp
import numpy as np
import orjson
import pandas as pd
import yaml
from numpy.exceptions import ComplexWarning

import sax

from .s import reciprocal

T = TypeVar("T")

__all__ = [
    "clean_string",
    "copy_settings",
    "flatten_dict",
    "get_settings",
    "grouped_interp",
    "hash_dict",
    "load_netlist",
    "load_recursive_netlist",
    "maybe",
    "merge_dicts",
    "read",
    "reciprocal",  # re-exported here to not break gplugins
    "rename_params",
    "rename_ports",
    "replace_kwargs",
    "same_args_as",
    "unflatten_dict",
    "update_settings",
]


def maybe(
    func: Callable[..., T], /, exc: type[Exception] = Exception
) -> Callable[..., T | None]:
    """Create a safe version of a function that returns None on exceptions.

    Wraps a function to catch specified exceptions and return None instead of
    raising them. This is useful for optional operations or when you want to
    gracefully handle failures.

    Args:
        func: Function to wrap for safe execution.
        exc: Exception type(s) to catch. Defaults to Exception (catches all).

    Returns:
        Wrapped function that returns None when the specified exception occurs.

    Example:
        ```python
        # Safe division that returns None for division by zero
        safe_divide = maybe(lambda x, y: x / y, ZeroDivisionError)
        result = safe_divide(10, 0)  # Returns None instead of raising

        # Safe file reading
        safe_read = maybe(lambda f: open(f).read(), FileNotFoundError)
        content = safe_read("nonexistent.txt")  # Returns None
        ```
    """

    @wraps(func)
    def new_func(*args: Any, **kwargs: Any) -> T | None:  # noqa: ANN401
        try:
            return func(*args, **kwargs)
        except exc:
            return None

    return new_func


def copy_settings(settings: sax.Settings) -> sax.Settings:
    """Create a deep copy of a settings dictionary.

    Creates a deep copy of a settings dictionary to avoid unintended mutations
    of shared parameter dictionaries.

    Args:
        settings: Settings dictionary to copy.

    Returns:
        Deep copy of the input settings dictionary.

    Example:
        ```python
        original = {"wl": 1.55, "temp": 300, "nested": {"param": 1.0}}
        copied = copy_settings(original)
        copied["nested"]["param"] = 2.0  # Doesn't affect original
        ```
    """
    return deepcopy(settings)


def read(content_or_filename: str | Path | sax.IOLike) -> str:
    r"""Read content from string, file path, or file-like object.

    Flexible content reader that can handle string content directly, file paths,
    or file-like objects. Automatically detects the input type and reads accordingly.

    Args:
        content_or_filename: Content as string (if contains newlines), file path,
            or file-like object with read() method.

    Returns:
        Content as string.

    Example:
        ```python
        # Direct string content
        content = read("line1\\nline2")

        # From file path
        content = read("config.yaml")

        # From file-like object
        from io import StringIO

        content = read(StringIO("data"))
        ```
    """
    if isinstance(content_or_filename, str) and "\n" in content_or_filename:
        return content_or_filename

    if isinstance(content_or_filename, str | Path):
        return Path(content_or_filename).read_text()

    return content_or_filename.read()


def load_netlist(content_or_filename: str | Path | sax.IOLike) -> sax.Netlist:
    """Load a SAX netlist from YAML content or file.

    Parses YAML content to create a SAX netlist dictionary. The YAML should
    contain instances, connections, and ports sections.

    Args:
        content_or_filename: YAML content as string, file path, or file-like object.

    Returns:
        Parsed netlist dictionary.

    Example:
        ```python
        # Load from file
        netlist = load_netlist("circuit.yml")

        # Load from YAML string
        yaml_content = '''
        instances:
          wg1:
            component: waveguide
        ports:
          in: wg1,in
          out: wg1,out
        '''
        netlist = load_netlist(yaml_content)
        ```
    """
    return yaml.safe_load(read(content_or_filename))


def load_recursive_netlist(
    top_level_path: str | Path,
    ext: str = ".pic.yml",
) -> sax.RecursiveNetlist:
    """Load a SAX recursive netlist from a directory of YAML files.

    Recursively loads all YAML files with the specified extension from a directory
    to create a recursive netlist. Each file becomes a component in the recursive
    netlist, with the filename (without extension) as the component name.

    Args:
        top_level_path: Path to the top-level netlist file.
        ext: File extension to search for. Defaults to ".pic.yml".

    Returns:
        Recursive netlist dictionary mapping component names to their netlists.

    Example:
        ```python
        # Load all .pic.yml files in directory
        recnet = load_recursive_netlist("circuits/main.pic.yml")
        # Result: {"main": {...}, "component1": {...}, "component2": {...}}
        ```
    """
    top_level_path = Path(top_level_path)
    folder_path = top_level_path.parent

    def _net_name(path: Path) -> sax.Name:
        return clean_string(path.name.removesuffix(ext))

    recnet = {_net_name(top_level_path): load_netlist(top_level_path)}

    for path in folder_path.rglob(ext):
        recnet[_net_name(path)] = load_netlist(path)

    return recnet


def clean_string(
    s: str, dot: str = "p", minus: str = "m", other: str = "_"
) -> sax.Name:
    """Clean a string to create a valid Python identifier.

    Converts arbitrary strings to valid Python identifiers by replacing special
    characters and ensuring the result starts with a letter or underscore.

    Args:
        s: String to clean.
        dot: Replacement for dots. Defaults to "p".
        minus: Replacement for minus/dash. Defaults to "m".
        other: Replacement for other special characters. Defaults to "_".

    Returns:
        Valid Python identifier string.

    Raises:
        ValueError: If cleaning fails to produce a valid identifier.

    Example:
        ```python
        clean_string("my-component.v2")  # Result: "my_componentpv2"
        clean_string("2stage_amp")  # Result: "_2stage_amp"
        clean_string("TE-mode")  # Result: "TEm_mode"
        ```
    """
    s = s.strip()
    s = s.replace(".", dot)  # dot
    s = s.replace("-", minus)  # minus
    s = re.sub("[^0-9a-zA-Z_]", other, s)
    if s[0] in "0123456789":
        s = "_" + s
    if not s.isidentifier():
        msg = f"failed to clean string to a valid python identifier: {s}"
        raise ValueError(msg)
    return s


def get_settings(model: sax.Model | sax.ModelFactory) -> sax.Settings:
    """Extract default parameter settings from a SAX model function.

    Inspects a model function's signature to extract default parameter values.
    This is useful for understanding what parameters a model accepts and their
    default values. Also supports Unpack[TypedDict] annotations on **kwargs.

    Args:
        model: SAX model function or model factory to inspect.

    Returns:
        Dictionary of parameter names and their default values.

    Example:
        ```python
        def my_model(wl=1.55, length=10.0, neff=2.4):
            return some_s_matrix


        settings = get_settings(my_model)
        # Result: {"wl": 1.55, "length": 10.0, "neff": 2.4}
        ```
    """
    signature = inspect.signature(model)
    settings: sax.Settings = {}

    for k, v in signature.parameters.items():
        # Handle regular parameters with defaults
        if v.default is not inspect.Parameter.empty:
            settings[k] = v.default
            continue

        # Handle **kwargs with Unpack[TypedDict]
        if (
            v.kind == inspect.Parameter.VAR_KEYWORD
            and v.annotation is not inspect.Parameter.empty
        ):
            origin = get_origin(v.annotation)
            if origin is Unpack:
                args = get_args(v.annotation)
                if args and len(args) > 0:
                    typed_dict_cls = args[0]
                    # Check if it's a TypedDict by looking for __annotations__
                    if hasattr(typed_dict_cls, "__annotations__"):
                        annotations = typed_dict_cls.__annotations__
                        for field_name in annotations:
                            # Check if there's a default value (Python 3.13+)
                            try:
                                default_value = getattr(typed_dict_cls, field_name)
                                # Verify it's not a method or function
                                if not inspect.ismethod(
                                    default_value
                                ) and not inspect.isfunction(default_value):
                                    settings[field_name] = default_value
                            except AttributeError:
                                # Field has no default value
                                pass

    return cast(sax.Settings, settings)


def merge_dicts(*dicts: dict) -> dict:
    """Merge multiple dictionaries with support for nested merging.

    Recursively merges multiple dictionaries, with later dictionaries
    taking precedence over earlier ones. Nested dictionaries are merged
    recursively rather than replaced entirely.

    Args:
        *dicts: Variable number of dictionaries to merge.

    Returns:
        Merged dictionary with nested structures preserved.

    Example:
        ```python
        dict1 = {"a": 1, "nested": {"x": 10}}
        dict2 = {"b": 2, "nested": {"y": 20}}
        merged = merge_dicts(dict1, dict2)
        # Result: {"a": 1, "b": 2, "nested": {"x": 10, "y": 20}}
        ```
    """
    num_dicts = len(dicts)

    if num_dicts < 1:
        return {}

    if num_dicts == 1:
        return dict(_generate_merged_dict(dicts[0], {}))

    if len(dicts) == 2:
        return dict(_generate_merged_dict(dicts[0], dicts[1]))

    return merge_dicts(dicts[0], merge_dicts(*dicts[1:]))


def replace_kwargs(func: Callable, **kwargs: sax.SettingsValue) -> None:
    """Change the kwargs signature of a function."""
    sig = inspect.signature(func)
    settings = [
        inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
        for k, v in kwargs.items()
    ]
    func.__signature__ = sig.replace(parameters=settings)  # type: ignore # noqa: PGH003


def update_settings(
    settings: sax.Settings, *compnames: str, **kwargs: sax.SettingsValue
) -> sax.Settings:
    """Update a nested settings dictionary with new parameter values.

    Updates settings for specific components or globally. Supports nested
    parameter dictionaries and selective component updates.

    Args:
        settings: Original settings dictionary to update.
        *compnames: Component names to update. If empty, updates all components.
        **kwargs: Parameter values to update.

    Returns:
        Updated settings dictionary (does not modify original).

    Note:
        - This operation never updates the given settings dictionary in place.
        - Any non-float keyword arguments will be silently ignored.
        - Even though it's possible to update parameter dictionaries in place,
          this function is convenient to apply certain parameters (e.g. wavelength
          'wl' or temperature 'T') globally.

    Example:
        ```python
        settings = {
            "wg1": {"length": 10.0, "neff": 2.4},
            "wg2": {"length": 20.0, "neff": 2.4},
        }
        # Update all components
        updated = update_settings(settings, wl=1.55)

        # Update specific component
        updated = update_settings(settings, "wg1", length=15.0)
        ```
    """
    _settings = {}
    if not compnames:
        for k, v in settings.items():
            if isinstance(v, dict):
                _settings[k] = update_settings(v, **kwargs)
            elif k in kwargs:
                _settings[k] = _try_complex_float(kwargs[k])
            else:
                _settings[k] = _try_complex_float(v)
    else:
        for k, v in settings.items():
            if isinstance(v, dict):
                if k == compnames[0]:
                    _settings[k] = update_settings(v, *compnames[1:], **kwargs)
                else:
                    _settings[k] = v
            else:
                _settings[k] = _try_complex_float(v)
    return _settings


def flatten_dict(dic: dict[str, Any], sep: str = ",") -> dict[str, Any]:
    """Flatten a nested dictionary into a single-level dictionary.

    Converts a nested dictionary into a flat dictionary by concatenating
    nested keys with a separator.

    Args:
        dic: Nested dictionary to flatten.
        sep: Separator to use between nested keys. Defaults to ",".

    Returns:
        Flattened dictionary with concatenated keys.

    Example:
        ```python
        nested = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        flat = flatten_dict(nested)
        # Result: {"a,b": 1, "a,c,d": 2, "e": 3}
        ```
    """
    return _flatten_dict(dic, sep=sep)


def unflatten_dict(dic: dict[str, Any], sep: str = ",") -> dict[str, Any]:
    """Unflatten a dictionary by splitting keys and creating nested structure.

    Converts a flattened dictionary back to nested form by splitting keys
    on the separator and creating the nested hierarchy.

    Args:
        dic: Flattened dictionary to unflatten.
        sep: Separator used in the flattened keys. Defaults to ",".

    Returns:
        Nested dictionary with original structure restored.

    Example:
        ```python
        flat = {"a,b": 1, "a,c,d": 2, "e": 3}
        nested = unflatten_dict(flat)
        # Result: {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        ```
    """
    # from: https://gist.github.com/fmder/494aaa2dd6f8c428cede
    items = {}

    for k, v in dic.items():
        keys = k.split(sep)
        sub_items = items
        for ki in keys[:-1]:
            if ki in sub_items:
                sub_items = sub_items[ki]
            else:
                sub_items[ki] = {}
                sub_items = sub_items[ki]

        sub_items[keys[-1]] = v

    return items


def grouped_interp(
    wl: sax.FloatArray, wls: sax.FloatArray, phis: sax.FloatArray
) -> sax.FloatArray:
    """Perform grouped phase interpolation for optical phase data.

    Grouped interpolation is useful for interpolating phase values where each
    datapoint is doubled (very close together) to give an indication of the
    phase variation at that point. This is common in optical simulations where
    phase unwrapping is needed.

    Args:
        wl: Wavelength points where interpolation is desired.
        wls: Reference wavelength points (1D array).
        phis: Phase values at reference wavelengths (1D array).

    Returns:
        Interpolated phase values at the requested wavelengths.

    Warning:
        This interpolation is only accurate in the range [wls[0], wls[-2])
        (wls[-2] not included). Any extrapolation outside these bounds can
        yield unexpected results!

    Raises:
        ValueError: If wls or phis are not 1D arrays or have mismatched shapes.

    Example:
        ```python
        import jax.numpy as jnp

        # Reference phase data with grouped points
        wls_ref = jnp.array([1.50, 1.501, 1.55, 1.551, 1.60, 1.601])
        phis_ref = jnp.array([0.1, 0.11, 0.5, 0.51, 1.0, 1.01])

        # Interpolate at new wavelengths
        wl_new = jnp.linspace(1.51, 1.59, 10)
        phis_interp = grouped_interp(wl_new, wls_ref, phis_ref)
        ```
    """
    wl = jnp.asarray(wl)
    wls = jnp.asarray(wls)
    phis = jnp.asarray(phis) % (2 * jnp.pi)
    phis = jnp.where(phis > jnp.pi, phis - 2 * jnp.pi, phis)
    if wls.ndim != 1:
        msg = "grouped_interp: wls should be a 1D array"
        raise ValueError(msg)
    if phis.ndim != 1:
        msg = "grouped_interp: wls should be a 1D array"
        raise ValueError(msg)
    if wls.shape != phis.shape:
        msg = "grouped_interp: wls and phis shape does not match"
        raise ValueError(msg)
    return _grouped_interp(wl.reshape(-1), wls, phis).reshape(*wl.shape)


def rename_params(model: sax.Model, renamings: dict[str, str]) -> sax.Model:
    """Rename the parameters of a model function.

    Creates a new model with renamed parameters while preserving the original
    functionality and default values.

    Args:
        model: Model function to rename parameters for.
        renamings: Dictionary mapping old parameter names to new names.

    Returns:
        New model function with renamed parameters.

    Raises:
        ValueError: If multiple old names map to the same new name.

    Example:
        ```python
        def original_model(wavelength=1.55, eff_index=2.4):
            return some_s_matrix


        # Rename parameters to standard names
        renamed_model = rename_params(
            original_model, {"wavelength": "wl", "eff_index": "neff"}
        )
        # Now can call: renamed_model(wl=1.55, neff=2.4)
        ```
    """
    reversed_renamings = {v: k for k, v in renamings.items()}
    if len(reversed_renamings) < len(renamings):
        msg = "Multiple old names point to the same new name!"
        raise ValueError(msg)

    if callable(_model := cast(sax.Model, model)):

        @wraps(_model)
        def new_model(**settings: sax.SettingsValue) -> sax.SType:
            old_settings = {
                reversed_renamings.get(k, k): v for k, v in settings.items()
            }
            return _model(**old_settings)

        new_settings = {renamings.get(k, k): v for k, v in get_settings(_model).items()}
        _replace_kwargs(new_model, **new_settings)

        return cast(sax.Model, new_model)

    msg = "rename_params should be used to decorate a Model."
    raise ValueError(msg)


@overload
def rename_ports(S: sax.SDict, renamings: dict[str, str]) -> sax.SDict: ...


@overload
def rename_ports(S: sax.SCoo, renamings: dict[str, str]) -> sax.SCoo: ...


@overload
def rename_ports(S: sax.SDense, renamings: dict[str, str]) -> sax.SDense: ...


@overload
def rename_ports(S: sax.Model, renamings: dict[str, str]) -> sax.Model: ...


def rename_ports(
    S: sax.SType | sax.Model, renamings: dict[str, str]
) -> sax.SType | sax.Model:
    """Rename the ports of an S-matrix or model.

    Creates a new S-matrix or model with renamed ports while preserving
    all S-parameter values and relationships.

    Args:
        S: S-matrix in any format or model function to rename ports for.
        renamings: Dictionary mapping old port names to new names.

    Returns:
        S-matrix or model with renamed ports.

    Raises:
        ValueError: If the input type is not supported for port renaming.

    Example:
        ```python
        # Rename ports in S-matrix
        s_matrix = {("input", "output"): 0.9}
        renamed_s = rename_ports(s_matrix, {"input": "in", "output": "out"})
        # Result: {("in", "out"): 0.9}


        # Rename ports in model
        def original_model(wl=1.55):
            return {("input", "output"): 0.9}


        renamed_model = rename_ports(original_model, {"input": "in", "output": "out"})
        ```
    """
    if callable(model := S):

        @wraps(model)
        def new_model(**settings: sax.SettingsValue) -> sax.SType:
            return rename_ports(model(**settings), renamings)

        return cast(sax.Model, new_model)

    if isinstance(sdict := S, dict):
        return {(renamings[p1], renamings[p2]): v for (p1, p2), v in sdict.items()}

    if len(scoo := cast(sax.SCoo, S)) == 4:
        Si, Sj, Sx, ports_map = scoo
        ports_map = {renamings[p]: i for p, i in ports_map.items()}
        return Si, Sj, Sx, ports_map

    if len(sdense := cast(sax.SDense, S)) == 2:
        Sx, ports_map = sdense
        ports_map = {renamings[p]: i for p, i in ports_map.items()}
        return Sx, ports_map

    msg = f"Cannot rename ports for type {type(S)}"
    raise ValueError(msg)


def hash_dict(dic: dict) -> int:
    """Compute a hash value for a dictionary.

    Creates a deterministic hash of a dictionary that can contain NumPy arrays
    and nested structures. Useful for caching and change detection.

    Args:
        dic: Dictionary to hash.

    Returns:
        Integer hash value.

    Example:
        ```python
        settings = {"wl": 1.55, "length": 10.0}
        hash_val = hash_dict(settings)
        # Same settings will always produce the same hash
        ```
    """
    return int(
        md5(
            orjson.dumps(
                _numpyfy(dic), option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SORT_KEYS
            )
        ).hexdigest(),
        16,
    )


class Normalization(NamedTuple):
    """Normalization parameters for an array.

    Contains the mean and standard deviation values needed to normalize
    and denormalize arrays. Typically used for machine learning preprocessing.

    Attributes:
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
    """

    mean: sax.ComplexArray
    std: sax.ComplexArray


def normalization(x: sax.ComplexArray, axis: int | None = None) -> Normalization:
    """Calculate normalization parameters (mean and std) for an array.

    Computes the mean and standard deviation of an array along the specified
    axis for use in normalization operations.

    Args:
        x: Input array to compute normalization parameters for.
        axis: Axis along which to compute statistics. If None, computes over
            the entire array.

    Returns:
        Normalization object containing mean and standard deviation.

    Example:
        ```python
        import jax.numpy as jnp

        data = jnp.array([[1, 2, 3], [4, 5, 6]])
        norm_params = normalization(data, axis=0)
        # Computes mean and std along axis 0
        ```
    """
    if axis is None:
        return Normalization(x.mean(), x.std())
    return Normalization(x.mean(axis), x.std(axis))


def cartesian_product(*arrays: sax.ComplexArray) -> sax.ComplexArray:
    """Calculate the n-dimensional Cartesian product of input arrays.

    Creates all possible combinations of elements from the input arrays,
    useful for parameter sweeps and grid generation.

    Args:
        *arrays: Variable number of arrays to compute Cartesian product for.

    Returns:
        Array where each row is a unique combination of input elements.

    Example:
        ```python
        import jax.numpy as jnp

        x = jnp.array([1, 2])
        y = jnp.array([10, 20])
        product = cartesian_product(x, y)
        # Result: [[1, 10], [1, 20], [2, 10], [2, 20]]
        ```
    """
    ixarrays = jnp.ix_(*arrays)
    barrays = jnp.broadcast_arrays(*ixarrays)
    sarrays = jnp.stack(barrays, -1)
    product = sarrays.reshape(-1, sarrays.shape[-1])
    return product


def normalize(x: sax.ComplexArray, normalization: Normalization) -> sax.ComplexArray:
    """Normalize an array using provided normalization parameters.

    Applies z-score normalization: (x - mean) / std.

    Args:
        x: Array to normalize.
        normalization: Normalization parameters containing mean and std.

    Returns:
        Normalized array with zero mean and unit standard deviation.

    Example:
        ```python
        data = jnp.array([1, 2, 3, 4, 5])
        norm_params = normalization(data)
        normalized = normalize(data, norm_params)
        ```
    """
    mean, std = normalization
    return (x - mean) / std


def denormalize(x: sax.ComplexArray, normalization: Normalization) -> sax.ComplexArray:
    """Denormalize an array using provided normalization parameters.

    Reverses z-score normalization: x * std + mean.

    Args:
        x: Normalized array to denormalize.
        normalization: Normalization parameters containing mean and std.

    Returns:
        Denormalized array with original scale and offset.

    Example:
        ```python
        normalized_data = jnp.array([-1, 0, 1])  # normalized
        norm_params = Normalization(mean=3.0, std=2.0)
        original = denormalize(normalized_data, norm_params)
        # Result: [1, 3, 5]
        ```
    """
    mean, std = normalization
    return x * std + mean


def wide_to_tidy(df: pd.DataFrame, ports: Iterable[str] | None = None) -> pd.DataFrame:
    """Convert a wide-format S-parameter DataFrame to tidy format.

    Args:
        df: Wide-format DataFrame with columns for wavelength, real, and imaginary
            parts of S-parameters.
        ports: Optional list of port names. If not provided, ports will be
            automatically generated as "o1", "o2", etc.

    Returns:
        Tidy-format DataFrame with columns:
            - 'wl': Wavelength values.
            - 'port_in': Input port names.
            - 'port_out': Output port names.
            - 'mode_in': Input mode labels.
            - 'mode_out': Output mode labels.
            - 're': Real part of S-parameters.
            - 'im': Imaginary part of S-parameters.
    """
    df.columns = [re.sub("[^a-z0-9]", "", c.lower()) for c in df.columns]
    df_im = df[["wl", *[c for c in df.columns if "im" in c]]]
    df_im = df_im.melt(id_vars="wl", value_name="im")
    if df_im.shape[0] == 0:
        msg = (
            "Given wide dataframe columns does not seem have columns "
            "with a 'im' (imaginary) marker."
        )
        raise ValueError(msg)
    df = cast(pd.DataFrame, df[["wl", *[c for c in df.columns if "re" in c]]])
    df = df.melt(id_vars="wl", value_name="re")
    if df.shape[0] == 0:
        msg = (
            "Given wide dataframe columns does not seem have columns "
            "with a 're' (real) marker."
        )
        raise ValueError(msg)
    port_idxs = [_get_port_idx_combos(s) for s in df["variable"]]
    port_idxs = sorted({idx for idxs in port_idxs for idx in idxs})
    if not ports:
        ports = [f"o{i + 1}" for i in port_idxs]
    else:
        ports = list(ports)
        for idx in port_idxs:
            try:
                ports[idx]
            except KeyError as e:
                msg = (
                    f"Cannot map port index {idx} to a port name. "
                    "is the given list of ports long enough?"
                )
                raise ValueError(msg) from e
    port_combos = [_get_port_combos(s, ports) for s in df["variable"]]
    df["port_in"] = [_get_port(p) for p, _ in port_combos]
    df["mode_in"] = [_get_mode(p) for p, _ in port_combos]
    df["port_out"] = [_get_port(q) for _, q in port_combos]
    df["mode_out"] = [_get_mode(q) for _, q in port_combos]
    df["im"] = df_im["im"]
    df = cast(
        pd.DataFrame,
        df[["wl", "port_in", "port_out", "mode_in", "mode_out", "re", "im"]],
    )
    return df


def same_args_as(
    func: Callable[..., T],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Have the same signature as the wrapped function."""

    def decorator(wrapped_func: Callable[..., T]) -> Callable[..., T]:
        name = wrapped_func.__name__
        qualname = wrapped_func.__qualname__
        module = wrapped_func.__module__
        doc = wrapped_func.__doc__
        new_func = wraps(func)(wrapped_func)
        new_func.__name__ = name
        new_func.__qualname__ = qualname
        new_func.__module__ = module
        new_func.__doc__ = doc
        return new_func

    return decorator


def _generate_merged_dict(dict1: dict, dict2: dict) -> Iterator[tuple[Any, Any]]:
    # inspired by https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    keys = {
        **dict.fromkeys(dict1),
        **dict.fromkeys(dict2),
    }  # keep key order, values irrelevant
    for k in keys:
        if k in dict1 and k in dict2:
            v1, v2 = dict1[k], dict2[k]
            if isinstance(v1, dict) or isinstance(v2, dict):
                # note: we used to have only merging when both where dicts,
                # but this way is more flexible - if one is not a dict, we
                # just replace it with an empty dict and continue merging.
                v = dict(
                    _generate_merged_dict(
                        {} if not isinstance(v1, dict) else v1,
                        {} if not isinstance(v2, dict) else v2,
                    )
                )
            else:
                # If neither of the values is a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                v = v2
        elif k in dict1:
            v = dict1[k]
        else:  # k in dict2:
            v = dict2[k]

        if isinstance(v, dict):
            yield (k, {**v})  # shallow copy of dict
        else:
            yield (k, v)


def _try_complex_float(f: Any) -> Any:  # noqa: ANN401
    """Try converting an object to float, return unchanged object on fail."""
    with warnings.catch_warnings():
        warnings.filterwarnings(action="error", category=ComplexWarning)
        try:
            return jnp.asarray(f, dtype=float)
        except ComplexWarning:
            return jnp.asarray(f, dtype=complex)
        except (ValueError, TypeError):
            pass

    return f


def _flatten_dict(
    dic: dict[str, Any], sep: str = ",", *, frozen: bool = False, parent_key: str = ""
) -> dict[str, Any]:
    items = []
    for k, v in dic.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(
                _flatten_dict(v, sep=sep, frozen=frozen, parent_key=new_key).items()
            )
        else:
            items.append((new_key, v))

    return dict(items)


@partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
@jax.jit
def _grouped_interp(
    wl: float,  # 0D array (not-vmapped) ; 1D array (vmapped)
    wls: sax.FloatArray,  # 1D array
    phis: sax.FloatArray,  # 1D array
) -> sax.FloatArray:
    dphi_dwl = (phis[1::2] - phis[::2]) / (wls[1::2] - wls[::2])
    phis = phis[::2]
    wls = wls[::2]
    dwl = (wls[1:] - wls[:-1]).mean(0, keepdims=True)

    t = (wl - wls + 1e-5 * dwl) / dwl  # small offset to ensure no values are zero
    t = jnp.where(jnp.abs(t) < 1, t, 0)
    m0 = jnp.where(t > 0, size=1)[0]
    m1 = jnp.where(t < 0, size=1)[0]
    t = t[m0]
    wl0 = wls[m0]
    wl1 = wls[m1]
    phi0 = phis[m0]
    phi1 = phis[m1]
    dphi_dwl0 = dphi_dwl[m0]
    dphi_dwl1 = dphi_dwl[m1]
    _phi0 = phi0 - 0.5 * (wl1 - wl0) * (dphi_dwl0 * (t**2 - 2 * t) - dphi_dwl1 * t**2)
    _phi1 = phi1 - 0.5 * (wl1 - wl0) * (
        dphi_dwl0 * (t - 1) ** 2 - dphi_dwl1 * (t**2 - 1)
    )
    phis = jnp.arctan2(
        (1 - t) * jnp.sin(_phi0) + t * jnp.sin(_phi1),
        (1 - t) * jnp.cos(_phi0) + t * jnp.cos(_phi1),
    )
    return phis


def _replace_kwargs(func: Callable, **kwargs: Any) -> None:  # noqa: ANN401
    """Change the kwargs signature of a function."""
    sig = inspect.signature(func)
    settings = [
        inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
        for k, v in kwargs.items()
    ]
    func.__signature__ = sig.replace(parameters=settings)  # type: ignore[reportFunctionMemberAccess]


def _numpyfy(obj: Any) -> Any:  # noqa: ANN401
    if not isinstance(obj, dict):
        return np.asarray(obj)
    return {k: _numpyfy(v) for k, v in obj.items()}


def _get_port_idx_combos(s: str) -> tuple[int, int]:
    match1, match2, *_ = list(re.finditer("[0-9]", s))
    idx1 = int(match1.group()) - 1
    idx2 = int(match2.group()) - 1
    return idx1, idx2


def _get_port_combos(s: str, ports: list[str]) -> tuple[str, str]:
    i, j = _get_port_idx_combos(s)
    return ports[i], ports[j]


def _get_port(pm: str) -> str:
    return pm.split("@")[0]


def _get_mode(pm: str) -> str:
    _, *m = pm.split("@")
    return "".join(m) or "1"
