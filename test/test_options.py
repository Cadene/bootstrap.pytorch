from bootstrap.lib.options import Options
from pprint import pprint

if __name__ == '__main__':
    # python test/lib_options.py
    # python test/lib_options.py --path_opts test/default.yaml
    # python test/lib_options.py --path_opts test/adam.yaml
    print(Options())

    Options().save('test/saved.yaml')

    print('-'*60)
    pprint(Options.load_yaml_opts('test/default.yaml'))
    print('-'*60)
    pprint(Options.load_yaml_opts('test/adam.yaml'))
    print('-'*60)
    pprint(Options.load_yaml_opts('test/saved.yaml'))


    print(Options()['message'])
    print('message' in Options())
