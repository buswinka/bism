import logging
import os
import os.path
import os.path
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

    if not isinstance(base_path, str):
        logging.error(f"bism.utils.save.return_script_text() kwarg base_path expects a string, not {type(base_path)}. {base_path=} -- Returning an Empty Dict")
        return {}

    if not os.path.exists(base_path):
        logging.error(f"bism.utils.save.return_script_text() kwarg {base_path=} does not exist. Returning an Empty Dict")
        return {}


    try:
        dir = {}
        for f in os.listdir(base_path):
            if any([f.endswith(e) for e in ext]):
                with open(f) as file:
                    text = file.read()
                    dir[f] = text

            elif os.path.isdir(f):
                dir[f] = return_script_text(f, ext)

        return dir

    except Exception as e:
        logging.critical(f'unkown error {e}. returning. ')
        return {}
