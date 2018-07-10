"""
# Load options using static method (no singleton)

- cli: `python test/test_options4.py`
- expected behavior:
```
OptionsDict({'message': 'default'})
OptionsDict({'message': 'adam', 'sgd': True, 'nested': OptionsDict({'message': 'lol'}), 'adam': True})
OptionsDict({'message': 'sgd', 'nested': OptionsDict({'message': 'save'}), 'sgd': True})
```
"""
from bootstrap.lib.options import Options

if __name__ == '__main__':
    print(Options.load_yaml_opts('test/default.yaml'))
    print(Options.load_yaml_opts('test/adam.yaml'))
    print(Options.load_yaml_opts('test/saved.yaml'))
