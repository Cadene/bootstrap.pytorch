"""
# Load Options

- cli: `python test/test_options3.py -o test/saved.yaml`
- expected behavior:
```
{
  "path_opts": "test/saved.yaml",
  "message": "sgd",
  "nested": {
    "message": "save"
  },
  "sgd": true
}
```
"""
from bootstrap.lib.options import Options

if __name__ == '__main__':
    print(Options())

