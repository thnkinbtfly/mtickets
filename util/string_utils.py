import re

def natural_keys(text):
    # from https://stackoverflow.com/a/5967539
    return [atoi(c) for c in re.split('(\d+)', text)]


def atoi(text):
    return int(text) if text.isdigit() else text


def add_pre_suf_to_keys_of(dict, prefix="", suffix=""):
    result = {}
    for name, value in dict.items():
        result[str(prefix + name + suffix)] = value
    return result


def grab_str_between_pre_suf(string, prefix, suffix):
    assert isinstance(string, str)
    left = string.find(prefix)
    if left < 0:
        return ''
    left += len(prefix)
    right = string.find(suffix, left)
    if right < 0:
        return ''
    return string[left:right]
