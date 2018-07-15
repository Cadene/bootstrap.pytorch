import os
import sys
import yaml
import json
import copy
import argparse
import collections
from collections import OrderedDict
from yaml import Dumper

def merge_dictionaries(dict1, dict2):
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            merge_dictionaries(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]


class OptionsDict(OrderedDict):

    def __init__(self, *args, **kwargs):
        super(OptionsDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key in self:
            val = OrderedDict.__getitem__(self, key)
        elif '.' in key:
            keys = key.split('.')
            val = self[keys[0]]
            for k in keys[1:]:
                val = val[k]
        else:
            OrderedDict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
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

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            OrderedDict.__getattr__(self, key)

    def __setattr__(self, key, value):
        self[key] = value

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '{}({})'.format(type(self).__name__, dictrepr)

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


# Options is a singleton
# https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
class Options(object):

    # Attributs

    __instance = None # singleton
    options = None # dictionnary of the singleton
    path_yaml = None

    class HelpParser(argparse.ArgumentParser):
        def error(self, message):
            print('\nError: %s\n' % message)
            self.print_help()
            sys.exit(2)

    # Build the singleton as Options()

    def __new__(self, path_yaml=None, arguments_callback=None):
        if not Options.__instance:
            Options.__instance = object.__new__(Options)

            fullopt_parser = Options.HelpParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

            if path_yaml:
                #Options.__instance.options = Options.load_yaml_opts(path_yaml)
                self.path_yaml = path_yaml
            else:
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

        return Options.__instance


    def __getitem__(self, key):
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

    # def update(self, *args, **kwargs):
    #     return self.options.update(*args, **kwargs)

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
            #try:
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
                #datatype = str if value is None else type(value[0]) if isinstance(value, list) else type(value)
                parser.add_argument(argname, help='Default: %(default)s', default=value, nargs=nargs, type=datatype)
            # except:
            #     import ipdb; ipdb.set_trace()


    def str_to_bool(self, v):
        true_strings = ['yes', 'true']#, 't', 'y', '1')
        false_strings = ['no', 'false']#, 'f', 'n', '0')
        if isinstance(v, str):
            if v.lower() in true_strings:
                return True
            elif v.lower() in false_strings:
                return False
        raise argparse.ArgumentTypeError('{} cant be converted to bool ('.format(v)+'|'.join(true_strings+false_strings)+' can be)')


    def save(self, path_yaml):
        Options.save_yaml_opts(self.options, path_yaml)

    # Static methods

    def load_yaml_opts(path_yaml):
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


if __name__ == '__main__':

    # # # # # # # # # # # # # # # #
    # Init OptionsDict from empty dict

    d = OptionsDict()
    d['a0'] = {'a1':'a2'}
    d['a0.b1'] = 'a2' # recursive setitem
    # d['a0.c1'] = 'a2' FAIL: must create dict first
    d.b0 = {} # setattr == setitem and {} converted into OptionsDict
    d.b0.a1 = 'a2'

    print(json.dumps(d, indent=2))
    # {
    #   "a0": {
    #     "a1": "b2"
    #   },
    #   "b0": {
    #     "a1": "a2"
    #   }
    # }

    print(d)
    # OptionsDict({'a0': OptionsDict({'a1': 'b2'}), 'b0': OptionsDict({'a1': 'a2'})})

    print(d['a0']['a1'])
    # a2

    print(d['a0.a1'])
    # a2

    print(d.a0.a1)
    # a2

    # # # # # # # # # # # # # # # #
    # Init OptionsDict from dict

    d = {'ha0':{'ha1':'ha2'}}
    print(json.dumps(d, indent=2))
    # {
    #   "ha0": {
    #     "ha1": "ha2"
    #   }
    # }
    print(d)
    # {'ha0': {'ha1': 'ha2'}}

    d = OptionsDict(d)
    print(json.dumps(d, indent=2))
    # {
    #   "ha0": {
    #     "ha1": "ha2"
    #   }
    # }
    print(d)
    # OptionsDict({'ha0': OptionsDict({'ha1': 'ha2'})})

    # # # # # # # # # # # # # # # #
    # Init Options

    # path = 'bootstrap/options/example.yaml'
    # Options(path)
    Options()
    print(json.dumps(Options().options, indent=2))

    print(Options()['dataset']['dir'])
    # /mnt/apcv_data/rcadene/data/cifar10
    print(Options()['dataset.dir'])
    # /mnt/apcv_data/rcadene/data/cifar10
    print(Options().dataset.dir)
    # /mnt/apcv_data/rcadene/data/cifar10
    print(Options().options['dataset']['dir'])
    # /mnt/apcv_data/rcadene/data/cifar10
    print(Options().options['dataset.dir'])
    # /mnt/apcv_data/rcadene/data/cifar10
    print(Options().options.dataset.dir)
    # /mnt/apcv_data/rcadene/data/cifar10
    Options().dataset.lol = 10
    print(Options()['dataset.lol'])
    # 10
    print(Options().options.dataset.lol)
    # 10
