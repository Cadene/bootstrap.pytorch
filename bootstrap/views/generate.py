from ..lib.options import Options
from ..run import main
from .factory import factory

def generate(path_opts=None):
    Options(path_yaml=path_opts)
    view = factory()
    view.generate()

if __name__ == '__main__':
    main(run=generate)