from .factory import factory
from ..lib.options import Options
from ..run import main


def generate(path_opts=None):
    Options(path_yaml=path_opts)
    view = factory()
    view.generate()


if __name__ == '__main__':
    main(run=generate)
