"""
# Test getter and setters

- cli: `python test/test_options1.py -o test/sgd.yaml`
- expected behavior:
```
l28: lol
l29: lol
l30: lol
l32: lol
l33: lol
l34: lol
l37: lolilol
l40: lolilolilol
l43: lolilolilolilol
```
"""
import inspect
from bootstrap.lib.options import Options

def pprint(*msg):
    caller_info = inspect.getframeinfo(inspect.stack()[1][0])
    print('l{}:'.format(caller_info.lineno), *msg)

if __name__ == '__main__':
    options = Options()

    pprint(options['nested']['message'])
    pprint(options['nested.message'])
    pprint(options.nested.message)

    pprint(Options()['nested']['message'])
    pprint(Options()['nested.message'])
    pprint(Options().nested.message)

    # Warning: modifying options is possible (but not desirable)

    Options()['nested.message'] = 'lolilol'
    pprint(Options().nested.message)

    Options().nested.message = 'lolilolilol'
    pprint(options.nested.message)

    options.nested.message = 'lolilolilolilol'
    pprint(Options()['nested']['message'])
