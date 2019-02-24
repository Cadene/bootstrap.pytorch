from .factory import factory
from ..run import main

def generate(path_opts=None):
    Options(path_opts=path_opts)
    view = factory()
    view.generate()

if __name__ == '__main__':
    main(run=generate)