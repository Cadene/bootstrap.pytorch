import os
import sys
import yaml
import json
import copy
import inspect
import argparse
import collections
from collections import OrderedDict
from yaml import Dumper
from bootstrap.lib.utils import merge_dictionaries


class OptionsDict(OrderedDict):
    """ Dictionary of options contained in the Options class
    """

    def __init__(self, *args, **kwargs):
        self.__locked = False
        super(OptionsDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if OrderedDict.__contains__(self, key):
            val = OrderedDict.__getitem__(self, key)
        elif '.' in key:
            keys = key.split('.')
            val = self[keys[0]]
            for k in keys[1:]:
                val = val[k]
        else:
            return OrderedDict.__getitem__(self, key)
        return val

    def __contains__(self, key):
        # Cannot use "key in" due to recursion, reusing rules for dotted keys from __getitem__
        try:
            self[key]
            return True
        except:
            return False

    def __setitem__(self, key, val):
        if key == '_{}__locked'.format(type(self).__name__):
            OrderedDict.__setitem__(self, key, val)
        elif hasattr(self, '_{}__locked'.format(type(self).__name__)):
            if self.__locked:
                raise PermissionError('Options\' dictionnary is locked and cannot be changed.')
            if type(val) == dict:
                val = OptionsDict(val)
                OrderedDict.__setitem__(self, key, val)
            elif '.' in key:
                keys = key.split('.')
                d = self[keys[0]]
                for k in keys[1:-1]:
                    d = d[k]
                d[keys[-1]] = val
            else:
                OrderedDict.__setitem__(self, key, val)
        else:
            raise PermissionError('Tried to access Options\' dictionnary bypassing the lock feature.')

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            return OrderedDict.__getattr__(self, key)

    # def __setattr__(self, key, value):
    #     self[key] = value

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '{}({})'.format(type(self).__name__, dictrepr)

    def get(self, key, default):
        if key in self:
            return self[key]
        else:
            return default


    def update(self, *args, **kwargs):
        for k, v in OrderedDict(*args, **kwargs).items():
            self[k] = v

    def asdict(self):
        d = {}
        for k, v in self.items():
            if isinstance(v, dict):
                d[k] = dict(v)
            else:
                d[k] = v
        return d

    def lock(self):
        self.__locked = True
        for key in self.keys():
            if type(key) == OptionsDict:
                self[key].lock()

    def islocked():
        return self.__locked

    def unlock(self):
        stack_this = inspect.stack()[1]
        stack_caller = inspect.stack()[2]
        if stack_this.filename != stack_caller.filename or stack_this.function != stack_caller.function:
            for i in range(10):
                print('WARNING: Options unlocked by {}[{}]: {}.'.format(
                    stack_caller.filename,
                    stack_caller.lineno,
                    stack_caller.function))
        self.__locked = False
        for key in self.keys():
            if type(key) == OptionsDict:
                self[key].unlock()

# https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
class Options(object):
    """ Options is a singleton. It parses a yaml file to generate rules to the argument parser.
        If a path to a yaml file is not provided, it relies on the `-o/--path_opts` command line argument.

        Args:
            path_yaml(str): path to the yaml file

        Example usage:
            
            .. code-block:: python

                # parse the yaml file and create options
                Options(path_yaml='bootstrap/options/example.yaml')
                
                opt = Options() # get the options dictionary from the singleton
                print(opt['exp'])     # display a dictionary
                print(opt['exp.dir']) # display a value
                print(opt['exp']['dir']) # display the same value
                
    """

    # Attributs

    __instance = None # singleton instance of this class
    options = None # dictionnary of the singleton
    path_yaml = None

    class HelpParser(argparse.ArgumentParser):
        def error(self, message):
            print('\nError: %s\n' % message)
            self.print_help()
            sys.exit(2)

    def __new__(self, path_yaml=None, arguments_callback=None, lock=False):
        # Options is a singleton, we will only build if it has not been built before
        if not Options.__instance:
            Options.__instance = object.__new__(Options)

            fullopt_parser = Options.HelpParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

            if path_yaml:
                self.path_yaml = path_yaml
            else:
                # Parsing only the path_opts argument to find yaml file
                optfile_parser = argparse.ArgumentParser(add_help=False)
                optfile_parser.add_argument('-o', '--path_opts', type=str, required=True)
                fullopt_parser.add_argument('-o', '--path_opts', type=str, required=True)
                self.path_yaml = optfile_parser.parse_known_args()[0].path_opts

            options_yaml = Options.load_yaml_opts(self.path_yaml)
            Options.__instance.add_options(fullopt_parser, options_yaml)

            arguments = fullopt_parser.parse_args()
            if arguments_callback:
                arguments = arguments_callback(Options.__instance, arguments, options_yaml)

            Options.__instance.options = OptionsDict()
            for argname in vars(arguments):
                nametree = argname.split('.')
                value = getattr(arguments, argname)

                position = Options.__instance.options
                for piece in nametree[:-1]:
                    if piece in position and isinstance(position[piece], collections.Mapping):
                        position = position[piece]
                    else:
                        position[piece] = {}
                        position = position[piece]
                position[nametree[-1]] = value

        if lock:
            Options.__instance.lock()
        return Options.__instance


    def __getitem__(self, key):
        """
        """
        val = self.options[key]
        return val


    def __setitem__(self, key, val):
        self.options[key] = val


    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            return object.__getattr__(self, key)


    def __contains__(self, item):
        return item in self.options


    def __str__(self):
        return json.dumps(self.options, indent=2)


    def get(self, key, default):
        return self.options.get(key, default)


    def copy(self):
        return self.options.copy()


    def has_key(self, k):
        return k in self.options


    def keys(self):
        return self.options.keys()


    def values(self):
        return self.options.values()


    def items(self):
        return self.options.items()


    def add_options(self, parser, options, prefix=''):
        if prefix:
            prefix += '.'

        for key, value in options.items():
            if isinstance(value, dict):
                self.add_options(parser, value, '{}{}'.format(prefix, key))
            else:
                argname = '--{}{}'.format(prefix, key)
                nargs = '*' if isinstance(value, list) else '?'
                if value is None:
                    datatype = str
                elif isinstance(value, bool):
                    datatype = self.str_to_bool
                elif isinstance(value, list):
                    if len(value) == 0:
                        datatype = str
                    else:
                        datatype = type(value[0])
                else:
                    datatype = type(value)
                parser.add_argument(argname, help='Default: %(default)s', default=value, nargs=nargs, type=datatype)


    def str_to_bool(self, v):
        true_strings = ['yes', 'true']
        false_strings = ['no', 'false']
        if isinstance(v, str):
            if v.lower() in true_strings:
                return True
            elif v.lower() in false_strings:
                return False
        raise argparse.ArgumentTypeError('{} cant be converted to bool ('.format(v)+'|'.join(true_strings+false_strings)+' can be)')


    def save(self, path_yaml):
        """ Write options dictionary to a yaml file
        """
        Options.save_yaml_opts(self.options, path_yaml)


    def lock(self):
        Options.__instance.options.lock()


    def unlock(self):
        Options.__instance.options.unlock()


    # Static methods


    def load_yaml_opts(path_yaml):
        """ Load options dictionary from a yaml file
        """
        # TODO: include the parent options when parsed, instead of after having loaded the main options
        result = {}
        with open(path_yaml, 'r') as yaml_file:
            options_yaml = yaml.load(yaml_file)
            includes = options_yaml.get('__include__', False)
            if includes:
                if type(includes) != list:
                    includes = [includes]
                for include in includes:
                    parent = Options.load_yaml_opts('{}/{}'.format(os.path.dirname(path_yaml), include))
                    merge_dictionaries(result, parent)
            merge_dictionaries(result, options_yaml) # to be sure the main options overwrite the parent options
        result.pop('__include__', None)
        result = OptionsDict(result)
        return result

    def save_yaml_opts(opts, path_yaml):
        # Warning: copy is not nested
        options = copy.copy(opts)
        if 'path_opts' in options:
            del options['path_opts']

        # https://gist.github.com/oglops/c70fb69eef42d40bed06
        def dict_representer(dumper, data):
            return dumper.represent_dict(data.items())
        Dumper.add_representer(OptionsDict, dict_representer)

        with open(path_yaml, 'w') as yaml_file:
            yaml.dump(options, yaml_file, Dumper=Dumper, default_flow_style=False)
