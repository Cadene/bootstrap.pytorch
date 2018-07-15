"""
# Merge two dictionnary and transform to dict

- cli: `python test/test_options5.py`
- expected behavior:
```
OptionsDict({'exp': OptionsDict({'dir': 'lol2', 'resume': None})})
{'exp': {'dir': 'lol2', 'resume': None}}
```
"""
from bootstrap.lib.options import merge_dictionaries
from bootstrap.lib.options import OptionsDict

if __name__ == '__main__':
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
    print(dict1)

    dict1 = dict1.asdict()
    print(dict1)