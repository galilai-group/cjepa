import json
import os
import pickle
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import glob
from os import path


def json_load(file):
    if isinstance(file, str):
        with open(file, 'r') as f:
            obj = json.load(f)
    elif hasattr(file, 'read'):
        obj = json.load(file)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def json_dump(obj, file=None, **kwargs):
    if file is None:
        return json.dumps(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'w') as f:
            json.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        json.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def yaml_load(file, **kwargs):
    kwargs.setdefault('Loader', Loader)
    if isinstance(file, str):
        with open(file, 'r') as f:
            obj = yaml.load(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = yaml.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def yaml_dump(obj, file=None, **kwargs):
    kwargs.setdefault('Dumper', Dumper)
    if file is None:
        return yaml.dump(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'w') as f:
            yaml.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        yaml.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def pickle_load(file, **kwargs):
    if isinstance(file, str):
        with open(file, 'rb') as f:
            obj = pickle.load(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = pickle.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def pickle_dump(obj, file=None, **kwargs):
    kwargs.setdefault('protocol', 2)
    if file is None:
        return pickle.dumps(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'wb') as f:
            pickle.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        pickle.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def load_obj(file, format=None, **kwargs):
    """Load contents from json/yaml/pickle files, and also supports custom
    arguments for each file format.

    This method provides a unified api for loading from serialized files.

    Args:
        file (str or file-like object): filename or the file-like object.
        format (None or str): if it is None, file format is inferred from the
            file extension, otherwise use the specified one. Currently
            supported formats are "json", "yaml", "yml", "pickle" and "pkl".

    Returns:
        Any: The content from the file.
    """
    processors = {
        'json': json_load,
        'yaml': yaml_load,
        'yml': yaml_load,
        'pickle': pickle_load,
        'pkl': pickle_load
    }
    if format is None and isinstance(file, str):
        format = file.split('.')[-1]
    if format not in processors:
        raise TypeError('Unsupported format: ' + format)
    return processors[format](file, **kwargs)


def dump_obj(obj, file=None, format=None, **kwargs):
    """Dump contents to json/yaml/pickle strings or files.

    This method provides a unified api for dumping to files, and also supports
    custom arguments for each file format.

    Args:
        file (None or str or file-like object): if None, then dump to a str,
            otherwise to a file specified by the filename or file-like object.
        obj (Any): the python object to be dumped
        format (None or str): same as :func:`load`

    Returns:
        bool: True for success, False otherwise
    """
    processors = {
        'json': json_dump,
        'yaml': yaml_dump,
        'yml': yaml_dump,
        'pickle': pickle_dump,
        'pkl': pickle_dump
    }
    if format is None:
        if isinstance(file, str):
            format = file.split('.')[-1]
        elif file is None:
            raise ValueError('format must be specified')
    if format not in processors:
        raise TypeError('Unsupported format: ' + format)
    return processors[format](obj, file, **kwargs)


def read_all_lines(file, strip=True):
    """Read all lines from a file."""
    with open(file, 'r') as f:
        lines = f.readlines()
    if strip:
        lines = [line.strip() for line in lines]
    return lines


def write_all_lines(lines, file):
    """Write all lines to a file."""
    with open(file, 'w') as f:
        f.writelines([line + '\n' for line in lines])


def strip_suffix(file):
    """Return the filename without suffix.
    E.g. 'xxx/video' for 'xxx/video.mp4'.

    Args:
        file (str): string to be processed.
    """
    assert isinstance(file, str)
    suffix = file.split('.')[-1]
    return file[:-(len(suffix) + 1)]


def check_file_exist(filename, msg_tmpl='file "{}" not exist:'):
    """Check whether a file exists."""
    if not path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def check_dir_exist(dirname, msg_tmpl='dir "{}" not exist:'):
    """Check whether a file exists."""
    if not path.isdir(dirname):
        raise FileNotFoundError(msg_tmpl.format(dirname))


def mkdir_or_exist(dir_name):
    """Create a new directory if not existed."""
    if not path.isdir(dir_name):
        os.makedirs(dir_name)


def scandir(dir_path, suffix=None):
    """Find all files under `dir_path` with some specific suffix."""
    if suffix is not None and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')
    for entry in os.scandir(dir_path):
        if not entry.is_file():
            continue
        filename = entry.name
        if suffix is None:
            yield filename
        elif filename.endswith(suffix):
            yield filename


def glob_all(dir_path, only_dir=False, sort=True, strip=True):
    """Similar to `scandir`, but return the entire path cat with dir_path."""
    if only_dir:
        pattern = path.join(dir_path, '*/')
    else:
        pattern = path.join(dir_path, '*')
    results = glob.glob(pattern)
    if sort:
        results.sort()
    if strip:
        results = [res.rstrip('/') for res in results]
    return results
