import os
import sys
import yaml
import pytest
from bootstrap.lib.options import Options
from bootstrap.lib.options import OptionsDict
from bootstrap.lib.utils import merge_dictionaries

def reset_options_instance():
    Options._Options__instance = None
    sys.argv = [sys.argv[0]] # reset command line args

def test_empty_path():
    """ Test empty path
  
        Expected behavior:

            .. code-block:: bash
                $ python tests/test_options.py
                usage: tests_options.py -o PATH_OPTS
                test_options.py: error: the following arguments are required: -o/--path_opts
    """
    reset_options_instance()
    try:
        Options()
        assert False
    except SystemExit as e:
        assert True

def test_o():
    """ Test path given in argument

        Expected behavior:

            .. code-block:: bash
                $ python tests/test_options.py -o test/default.yaml
                {
                  "path_opts": "test/default.yaml",
                  "message": "default"
                }
    """
    reset_options_instance()
    sys.argv += ['--path_opts', 'tests/default.yaml']
    assert (Options().options == OptionsDict({'path_opts': 'tests/default.yaml', 'message': 'default'}))

def test_path_opts():
    """ Test path given in argument

        Expected behavior:

            .. code-block:: bash
                $ python tests/test_options.py --path_opts test/default.yaml
                {
                  "path_opts": "test/default.yaml",
                  "message": "default"
                }
    """
    reset_options_instance()
    sys.argv += ['-o', 'tests/default.yaml']
    assert (Options().options == OptionsDict({'path_opts': 'tests/default.yaml', 'message': 'default'}))

def test_path_opts_h():
    """ Test path given in argument with help

        Expected behavior:

            .. code-block:: bash
                $ python tests/test_options.py -o test/default.yaml -h                
                usage: tests/test_options.py [-h] -o PATH_OPTS [--message [MESSAGE]]

                optional arguments:
                  -h, --help            show this help message and exit
                  -o PATH_OPTS, --path_opts PATH_OPTS
                  --message [MESSAGE]   Default: default
    """
    reset_options_instance()
    sys.argv += ['-o', 'tests/default.yaml', '-h']
    try:
        Options()
        assert False
    except SystemExit as e:
        assert True


def test_include():
    """ Test include

        Expected behavior:

            .. code-block:: bash
                $ python tests/test_options.py -o tests/sgd.yaml               
                {
                  "path_opts": "test/sgd.yaml",
                  "message": "sgd",
                  "sgd": true,
                  "nested": {
                    "message": "lol"
                  }
                }
    """
    reset_options_instance()
    sys.argv += ['-o', 'tests/sgd.yaml']
    assert (Options().options == OptionsDict({
        "path_opts": "tests/sgd.yaml",
        "message": "sgd",
        "sgd": True,
        "nested": {
            "message": "lol"
        }
    }))

def test_overwrite():
    """ Test overwrite

        Expected behavior:

            .. code-block:: bash
                $ python tests/test_options.py -o tests/sgd.yaml --nested.message lolilol`
                {
                    "path_opts": "tests/sgd.yaml",
                    "message": "sgd",
                    "sgd": true,
                    "nested": {
                        "message": "lolilol"
                    }
                }
    """
    reset_options_instance()
    sys.argv += ['-o', 'tests/sgd.yaml', '--nested.message', 'lolilol']
    assert (Options().options == OptionsDict({
        "path_opts": "tests/sgd.yaml",
        "message": "sgd",
        "sgd": True,
        "nested": {
            "message": "lolilol"
        }
    }))

def test_getters():
    """ Test getters
    """
    reset_options_instance()
    sys.argv += ['-o', 'tests/sgd.yaml']
    opt = Options()
    assert opt['nested']['message'] == 'lol'
    assert opt['nested.message'] == 'lol'
    assert opt.nested.message == 'lol'

# TODO: test_setters

def test_save():
    """ Test save and load
    """
    reset_options_instance()
    sys.argv += ['-o', 'tests/sgd.yaml', '--nested.message', 'save']
    path_yaml = 'tests/saved.yaml'
    Options().save(path_yaml)
    with open(path_yaml, 'r') as yaml_file:
        options_yaml = yaml.load(yaml_file)
    assert (OptionsDict(options_yaml) == OptionsDict({
        "message": "sgd",
        "sgd": True,
        "nested": {
            "message": "save"
        }
    }))
    reset_options_instance()
    sys.argv += ['-o', 'tests/saved.yaml']
    assert (Options().options == OptionsDict({
        "path_opts": "tests/saved.yaml",
        "message": "sgd",
        "sgd": True,
        "nested": {
            "message": "save"
        }
    }))

def test_load_yaml_opts():
    """ Load options using static method (no singleton)
    """
    reset_options_instance()
    opt = Options.load_yaml_opts('tests/default.yaml')
    assert (opt == OptionsDict({'message': 'default'}))
    assert Options._Options__instance is None

def test_merge_dictionaries():
    """ Merge two dictionnary
    """
    dict1 = {
        'exp': {
            'dir': 'lol1',
            'resume': None
        }
    }
    dict2 = {
        'exp': {
            'dir': 'lol2'
        }
    }
    dict1 = OptionsDict(dict1)
    dict2 = OptionsDict(dict2)
    merge_dictionaries(dict1, dict2)
    assert (dict1 == OptionsDict({'exp': OptionsDict({'dir': 'lol2', 'resume': None})}))

def test_as_dict():
    """ Copy OptionsDict in a new dictionary of type :mod:`dict`
    """
    dict1 = {
        'exp': {
            'dir': 'lol1',
            'resume': None
        }
    }
    assert (dict1 == OptionsDict(dict1).asdict())
