import argparse
import json
import os
import sys

from io import StringIO

import pytest

import yaml

from bootstrap.lib.options import Options, OptionsDict
from bootstrap.lib.utils import merge_dictionaries


def reset_options_instance():
    Options._Options__instance = None
    sys.argv = [sys.argv[0]]  # reset command line args


def test_empty_path(monkeypatch):
    """ Test empty path

        Expected behavior:

            .. code-block:: bash
                $ python tests/test_options.py
                usage: tests_options.py -o PATH_OPTS
                test_options.py: error: the following arguments are required: -o/--path_opts
    """
    reset_options_instance()

    monkeypatch.setattr(os, '_exit', sys.exit)

    with pytest.raises(Options.MissingOptionsException):
        Options()


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
    with pytest.raises(Options.MissingOptionsException):
        Options()


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
    assert Options().options == OptionsDict({
        "path_opts": "tests/sgd.yaml",
        "message": "sgd",
        "sgd": True,
        "nested": {
            "message": "lol"
        }
    })


def test_include_list():
    reset_options_instance()
    sys.argv += ['-o', 'tests/sgd_list_include.yaml']
    assert Options().options == OptionsDict({
        "path_opts": "tests/sgd_list_include.yaml",
        "message": "sgd",
        "sgd": True,
        "nested": {
            "message": "lol"
        },
        "database": "db",
    })


def test_include_absolute_path():
    reset_options_instance()
    path_file = os.path.join(os.getcwd(), 'tests', 'sgd_abs_include.yaml')
    include_file = os.path.join(os.getcwd(), 'tests', 'default.yaml')
    options = {
        '__include__': include_file,
        'sgd': True,
        'nested': {'message': 'lol'},
    }
    with open(path_file, 'w') as f:
        yaml.dump(options, f, default_flow_style=False)
    sys.argv += ['-o', 'tests/sgd_abs_include.yaml']
    gt_options = {
        "path_opts": 'tests/sgd_abs_include.yaml',
        "message": "default",
        "sgd": True,
        "nested": {
            "message": "lol"
        }
    }
    assert Options().options.asdict() == gt_options
    os.remove(path_file)


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
        options_yaml = yaml.safe_load(yaml_file)
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


def test_initialize_options_source_dict_1():
    reset_options_instance()
    source = {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    Options(source, run_parser=False)
    assert Options().options == OptionsDict(source)
    assert Options().source == source


def test_initialize_options_source_dict_2():
    reset_options_instance()
    sys.argv += ['-o', 'tests/default.yaml', '--model.network', 'mynet']
    source = {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    Options(source, run_parser=True)
    assert Options()['model']['network'] == 'mynet'


def test_initialize_options_source_dict_3():
    reset_options_instance()
    source1 = {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    Options(source1, run_parser=False)
    assert Options().options == OptionsDict(source1)
    assert Options().source == source1

    source2 = {
        'Micael': 'is the best',
        'Remi': 'is awesome',
    }
    Options(source2, run_parser=False)
    assert Options().options == OptionsDict(source1)
    assert Options().source == source1


def test_initialize_options_source_dict_4():
    reset_options_instance()
    source = {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    with pytest.raises(Options.MissingOptionsException):
        Options(source, run_parser=True)


def test_initialize_options_source_optionsdict():
    reset_options_instance()
    source = OptionsDict({
        'dataset': 124,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    })
    Options(source, run_parser=False)
    assert Options().options == source
    assert Options().source == source.asdict()


def test_initialize_options_incorrect_source():
    reset_options_instance()
    source = 123
    with pytest.raises(TypeError):
        Options(source, run_parser=False)


def test_initialize_arguments_callback():
    reset_options_instance()
    sys.argv += ['-o', 'tests/default.yaml']
    source = {
        'dataset': 'mydataset',
        'model': 'mymodel',
    }

    def arguments_callback_a(instance, arguments, options_dict):
        arguments.dataset = arguments.dataset + 'a'
        arguments.model = arguments.model + 'a'
        return arguments

    Options(source, arguments_callback=arguments_callback_a)
    source_a = {
        'path_opts': 'tests/default.yaml',
        'dataset': 'mydataseta',
        'model': 'mymodela',
    }
    assert Options().options == OptionsDict(source_a)
    assert Options().source == source


def test_initialize_lock():
    reset_options_instance()
    source = {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    Options(source, run_parser=False, lock=True)
    assert Options().options.islocked()


def test_initialize_not_locked():
    reset_options_instance()
    source = {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    Options(source, run_parser=False, lock=False)
    assert not Options().options.islocked()


def test_setitem_1():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    assert Options().options == source
    Options()['abc'] = 'new value'
    assert Options()['abc'] == 'new value'


def test_setitem_2():
    reset_options_instance()
    source = {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    Options(source, run_parser=False)
    assert Options().options == source
    Options()['model.criterion'] = 'new value'
    assert Options()['model.criterion'] == 'new value'


def test_setitem_3():
    reset_options_instance()
    source = {'dataset': 123}
    Options(source, run_parser=False)
    Options()['model.criterion'] = 'I am a criterion'
    Options()['model.network'] = 'I am a network'
    assert Options().options == {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    assert Options()['model.criterion'] == 'I am a criterion'
    assert Options()['model.network'] == 'I am a network'


def test_setitem_4():
    reset_options_instance()
    source = {
        'model.name': 'model',
    }
    Options(source, run_parser=False)
    Options()['model.topk'] = [1, 2]
    Options()['model.network.input_size'] = 12
    Options()['model.network.output_size'] = 4
    Options()['dataset.dataloader.batch_size'] = 8
    assert Options().options == {
        'dataset': {'dataloader': {'batch_size': 8}},
        'model': {
            'name': 'model',
            'topk': [1, 2],
            'network': {
                'input_size': 12,
                'output_size': 4
            }
        },
    }


def test_setitem_key_int():
    reset_options_instance()
    source = {1: 123}
    Options(source, run_parser=False)
    assert Options().options == source
    Options()[1] = 'new value'
    assert Options()[1] == 'new value'


def test_setitem_key_float():
    reset_options_instance()
    source = {1.2: 123}
    Options(source, run_parser=False)
    assert Options().options == source
    Options()[1.2] = 'new value'
    assert Options()[1.2] == 'new value'


def test_setitem_key_bytes():
    reset_options_instance()
    source = {bytes(1): 123}
    Options(source, run_parser=False)
    assert Options().options == source
    Options()[bytes(2)] = 'new value'
    assert Options()[bytes(2)] == 'new value'


def test_getattr():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    assert Options().options == source
    assert Options().abc == 123


def test_get_exist_value():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    assert Options().options == source
    value = Options().get('abc', 'default value')
    assert value == 123


def test_get_default_value():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    assert Options().options == source
    value = Options().get('cba', 'default value')
    assert value == 'default value'


def test_has_key_true():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    assert Options().options == source
    assert 'abc' in Options()


def test_has_key_false():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    assert Options().options == source
    assert 'cba' not in Options()


def test_keys():
    reset_options_instance()
    source = {
        'model': 'mymodel',
        'dataset': 'mydataset'
    }
    Options(source, run_parser=False)
    assert Options().options == source
    assert sorted(Options().keys()) == sorted(['model', 'dataset'])


def test_values():
    reset_options_instance()
    source = {
        'model': 'mymodel',
        'dataset': 'mydataset'
    }
    Options(source, run_parser=False)
    assert Options().options == source
    assert sorted(Options().values()) == sorted(['mymodel', 'mydataset'])


def test_items():
    reset_options_instance()
    source = {'model': 'mymodel'}
    Options(source, run_parser=False)
    assert Options().options == source
    for key, value in Options().items():
        assert key == 'model'
        assert value == 'mymodel'


def test_lock():
    reset_options_instance()
    source = {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    Options(source, run_parser=False)
    assert Options().options == source
    Options().unlock()
    assert not Options().options.islocked()
    assert not Options().options['model'].islocked()
    Options().lock()
    assert Options().options.islocked()
    assert Options().options['model'].islocked()


def test_unlock():
    reset_options_instance()
    source = {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    Options(source, run_parser=False)
    assert Options().options == source
    Options().lock()
    assert Options().options.islocked()
    assert Options().options['model'].islocked()

    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result

    Options().unlock()

    sys.stdout = old_stdout

    assert not Options().options.islocked()
    assert not Options().options['model'].islocked()

    result_string = result.getvalue()

    # Should print more than 3 times
    assert len(result_string.splitlines()) > 3


def test_lock_setitem():
    reset_options_instance()
    source = {
        'dataset': 123,
        'model': {
            'criterion': 'I am a criterion',
            'network': 'I am a network',
        },
    }
    Options(source, run_parser=False)
    assert Options().options == source
    Options().lock()
    with pytest.raises(PermissionError):
        Options()['dataset'] = 421


def test_str_to_bool_yes():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    assert Options().str_to_bool('yes')
    assert Options().str_to_bool('Yes')
    assert Options().str_to_bool('YES')


def test_str_to_bool_true():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    assert Options().str_to_bool('true')
    assert Options().str_to_bool('True')
    assert Options().str_to_bool('TRUE')


def test_str_to_bool_no():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    assert not Options().str_to_bool('no')
    assert not Options().str_to_bool('No')
    assert not Options().str_to_bool('NO')


def test_str_to_bool_false():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    assert not Options().str_to_bool('false')
    assert not Options().str_to_bool('False')
    assert not Options().str_to_bool('FALSE')


def test_str_to_bool_incorrect():
    reset_options_instance()
    source = {'abc': 123}
    Options(source, run_parser=False)
    with pytest.raises(argparse.ArgumentTypeError):
        Options().str_to_bool('incorrect')


def test_str():
    reset_options_instance()
    source = {'abc': 123, 'key1': 'value1'}
    Options(source, run_parser=False)
    assert Options().options == source
    str_representation = Options().__str__()
    opt_dict = json.loads(str_representation)
    assert isinstance(str_representation, str)
    assert opt_dict == source


def test_add_options():
    reset_options_instance()
    sys.argv += [
        '-o', 'tests/default.yaml',
        '--dataset', '421',
        '--value', '2',
        '--model.metric', 'm1', 'm2',
    ]
    source = {
        'dataset': 123,
        'value': 1.5,
        'model': {
            'criterion': ['mse', 'l1'],
            'network': 'I am a network',
            'metric': [],
        },
        'useless': None,
    }
    Options(source, run_parser=True)
    assert Options()['dataset'] == 421
    assert Options()['value'] == 2
    assert isinstance(Options()['value'], float)
    assert Options()['model']['metric'] == ['m1', 'm2']
