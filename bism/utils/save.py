import torch
import glob

import os.path
import os
from typing import List


def return_script_text(base_path: str, ext: str | List[str] = '.py'):
    """
    recursivly searches all py files in a dir and returns a dict of all while maintaining the tree

    :param base_path: path to base folder
    :param ext: extension of extension
    :return:
    """
    if not isinstance(ext, list):
        ext = [ext]

    dir = {}
    for f in os.listdir(base_path):
        if any([f.endswith(e) for e in ext]):
            with open(f) as file:
                text = file.read()
                dir[f] = text

        elif os.path.isdir(f):
            dir[f] = return_script_text(f, ext)

    return dir
