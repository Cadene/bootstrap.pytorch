"""
# Save Options

- cli: `python test/test_options2.py -o test/sgd.yaml --nested.message save`
- expected behavior: overwrite nested.message and save options in file
"""
from bootstrap.lib.options import Options

if __name__ == '__main__':
    Options().save('test/saved.yaml')
