from ..lib.logger import Logger
from .engine import Engine

class LoggerEngine(Engine):

    def __init__(self):
        super(LoggerEngine, self).__init__()

    def is_best(self, out, saving_criteria):
        if ':lt' in saving_criteria:
            name = saving_criteria.replace(':lt', '') # less than
            order = '<' #first_best = float('+inf')
        elif ':gt' in saving_criteria:
            name = saving_criteria.replace(':gt', '') # greater than
            order = '>' #first_best = float('-inf')
        else:
            error_msg = """'--engine.saving_criteria' named '{}' does not specify order
            by ending with a ':lt' (new best value < last best value) or 
            ':gt' (resp. >) to specify order""".format(saving_criteria)
            raise ValueError(error_msg)

        if name in out:
            new_value = out[name]
        elif name in Logger().values:
            new_value = Logger().values[name][-1]
        else:
            raise ValueError("name '{}' not in outputs '{}' and not in logger '{}'".format(
                name, list(out.keys()), list(Logger().values.keys())))

        if name not in self.best_out:
            self.best_out[name] = new_value
        else:
            if eval('{} {} {}'.format(new_value, order, self.best_out[name])):
                self.best_out[name] = new_value
                return True

        return False

