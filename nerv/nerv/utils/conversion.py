def list_cast(in_list, dst_type):
    """Convert a list of items to some type.

    Args:
        in_list (List[Any]): the list to be casted.
        dst_type (Type): the target type to convert to.

    Returns:
        List[`dst_type`]: converted list of target type.
    """
    if not isinstance(in_list, list):
        raise TypeError('"in_list" must be a list')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')
    return list(map(dst_type, in_list))


def is_list_of(in_list, expected_type):
    """Check whether it is a list of objects of a certain type.

    Args:
        in_list (List[Any]): the list to be checked.
        expected_type (Type): the expected type of list items.

    Returns:
        bool: check result.
    """
    if not isinstance(in_list, list):
        return False
    for item in in_list:
        if not isinstance(item, expected_type):
            return False
    return True


def slice_list(in_list, lens):
    """Slice a list into several sub lists by a list of given length.

    Args:
        in_list (List[Any]): the list to be sliced.
        lens (List[int]): the expected length of each out list.

    Returns:
        List[List[Any]]: list of sliced list.
    """
    if not isinstance(lens, list) or not is_list_of(lens, int):
        raise TypeError('"indices" must be a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError('list length and summation of lens do not match')
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def merge_list(in_list):
    """Merge a list of list into a single list.

    Args:
        in_list (List[List[Any]]): the list of list to be merged.

    Returns:
        List[Any]: the flatten list.
    """
    out_list = []
    for sub_list in in_list:
        out_list.extend(sub_list)
    return out_list


def list2str(lst):
    """Convert [a, b, ..., x] to 'a_b_..._x'."""
    return '_'.join(str(item) for item in lst)


def str2bool(var):
    """Convert string value into boolean value. Used in argparse flag.

    Args:
        var (str): string to convert.

    Returns:
        bool: boolean value.
    """
    return var.lower() in ("yes", "true", "t", "1")
