"""
# Test empty path

- cli: `python test/test_options0.py`
- expected behavior:
```
usage: test_options0.py -o PATH_OPTS
test_options0.py: error: the following arguments are required: -o/--path_opts
```

###########################################################

# Test path given in argument

- cli: `python test/test_options0.py --path_opts test/default.yaml`
- expected behavior:
```
{
  "path_opts": "test/default.yaml",
  "message": "default"
}
```

###########################################################

# Test path given in argument with help

- cli: `python test/test_options0.py --path_opts test/default.yaml -h`
- expected behavior:
```
usage: test_options0.py [-h] -o PATH_OPTS [--message [MESSAGE]]

optional arguments:
  -h, --help            show this help message and exit
  -o PATH_OPTS, --path_opts PATH_OPTS
  --message [MESSAGE]   Default: default
```

###########################################################

# Test include

- cli: `python test/test_options0.py -o test/sgd.yaml`
- expected behavior:
```
{
  "path_opts": "test/sgd.yaml",
  "message": "sgd",
  "sgd": true,
  "nested": {
    "message": "lol"
  }
}
```

###########################################################

# Test overwrite

- cli: `python test/test_options0.py -o test/sgd.yaml --nested.message lolilol`
- expected behavior:
```
{
  "path_opts": "test/sgd.yaml",
  "message": "sgd",
  "sgd": true,
  "nested": {
    "message": "lolilol"
  }
}
```
"""


from bootstrap.lib.options import Options

if __name__ == '__main__':
    options = Options()
    print(options)
