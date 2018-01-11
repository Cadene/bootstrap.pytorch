import os
import sys
import yaml
import copy
import argparse
import collections
from .utils import merge_dictionaries

# Options is a singleton
# https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

class Options(object):

    # Attributs

    __instance = None
    options = None

    class HelpParser(argparse.ArgumentParser):
        def error(self, message):
            print('\nError: %s\n' % message)
            self.print_help()
            sys.exit(2)

    # Methods

    def __new__(self, path_yaml=None, arguments_callback=None):
        if not Options.__instance:
            Options.__instance = object.__new__(Options)

            if path_yaml:
                Options.__instance.options = Options.__instance.load_yaml_opts(path_yaml)

            else:
                try:
                    optfile_parser = argparse.ArgumentParser(add_help=False)
                    fullopt_parser = Options.HelpParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

                    optfile_parser.add_argument('--path_opts', default='options/default.yaml', help='path to a yaml file containing the default options')
                    fullopt_parser.add_argument('--path_opts', default='options/default.yaml', help='path to a yaml file containing the default options')

                    options_yaml = Options.__instance.load_yaml_opts(optfile_parser.parse_known_args()[0].path_opts)
                    Options.__instance.add_options(fullopt_parser, options_yaml)

                    arguments = fullopt_parser.parse_args()
                    if arguments_callback:
                        arguments = arguments_callback(Options.__instance, arguments, options_yaml)

                    Options.__instance.options = {}
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
                    
                except:
                    Options.__instance = None
                    raise
        
        return Options.__instance.options
    
    def get_instance(self):
        return Options.__instance

    def load_from_yaml(path_yaml):
        Options.__instance = object.__new__(Options)
        Options.__instance.options = Options.__instance.load_yaml_opts(path_yaml)
        print('Options.load_from_yaml(path_yaml) is depreciated. Please use Options(path_yaml) instead.')

    def load_yaml_opts(self, path_yaml):
        result = {}
        with open(path_yaml, 'r') as yaml_file:
            options_yaml = yaml.load(yaml_file)
            if 'parent' in options_yaml.keys() and options_yaml['parent'] is not None:
                parent = self.load_yaml_opts('{}/{}'.format(os.path.dirname(path_yaml), options_yaml['parent']))
                merge_dictionaries(result, parent)
            merge_dictionaries(result, options_yaml)
        result.pop('parent', None)
        return result

    def save_yaml_opts(path_yaml):
        # Warning: copy is not nested
        options = copy.copy(Options.__instance.options)
        del options['path_opts']
        with open(path_yaml, 'w') as yaml_file:
            yaml.dump(options, yaml_file, default_flow_style=False)

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
