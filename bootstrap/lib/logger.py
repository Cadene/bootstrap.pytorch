#################################################################################
# By Micael Carvalho and Remi Cadene; https://github.com/MicaelCarvalho/logger  #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE #
# SOFTWARE.                                                                     #
#################################################################################

import collections
import datetime
import inspect
import numbers
import os
import sqlite3
import sys


class Logger(object):
    """ The Logger class is a singleton. It contains all the utilities
        for logging variables in a key-value dictionary.
        It can also be considered as a replacement for the print function.

        .. code-block:: python

            Logger(dir_logs='logs/mnist')
            Logger().log_value('train_epoch.epoch', epoch)
            Logger().log_value('train_epoch.mean_acctop1', mean_acctop1)
            Logger().flush() # write the logs.sqlite

            Logger()("Launching training procedures")  # written to logs.txt
            > [I 2018-07-23 18:58:31] ...trap/engines/engine.py.80: Launching training procedures
    """

    DEBUG = -1
    INFO = 0
    SUMMARY = 1
    WARNING = 2
    ERROR = 3
    SYSTEM = 4
    _instance = None
    indicator = {DEBUG: 'D', INFO: 'I', SUMMARY: 'S', WARNING: 'W', ERROR: 'E', SYSTEM: 'S'}

    class Colors:
        END = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        GREY = 30
        RED = 31
        GREEN = 32
        YELLOW = 33
        BLUE = 34
        PURPLE = 35
        SKY = 36
        WHITE = 37
        BACKGROUND = 10
        LIGHT = 60
        @staticmethod
        def code(value):
            '\033[{}m'.format(value)

    colorcode = {
        DEBUG: Colors.code(Colors.SKY),
        INFO: Colors.code(Colors.GREY + Colors.LIGHT),
        SUMMARY: Colors.code(Colors.BLUE + Colors.LIGHT),
        WARNING: Colors.code(Colors.YELLOW + Colors.LIGHT),
        ERROR: Colors.code(Colors.RED + Colors.LIGHT),
        SYSTEM: Colors.code(Colors.WHITE + Colors.LIGHT)
    }

    log_level = None  # log level
    dir_logs = None
    sqlite_cur = None
    sqlite_file = None
    sqlite_conn = None
    path_txt = None
    file_txt = None
    name = None
    max_lineno_width = 3

    def __new__(cls, dir_logs=None, name='logs'):
        if Logger._instance is None:
            Logger._instance = object.__new__(Logger)
            Logger._instance.set_level(Logger._instance.INFO)

            if dir_logs:
                Logger._instance.name = name
                Logger._instance.dir_logs = dir_logs
                Logger._instance.path_txt = os.path.join(dir_logs, '{}.txt'.format(name))
                Logger._instance.file_txt = open(os.path.join(dir_logs, '{}.txt'.format(name)), 'a+')
                Logger._instance.sqlite_file = os.path.join(dir_logs, '{}.sqlite'.format(name))
                Logger._instance.init_sqlite()
            else:
                Logger._instance.log_message('No logs files will be created (dir_logs attribute is empty)',
                                             log_level=Logger.WARNING)

        return Logger._instance

    def __call__(self, *args, **kwargs):
        return self.log_message(*args, **kwargs, stack_displacement=2)

    def set_level(self, log_level):
        self.log_level = log_level

    def log_message(self, *message, log_level=INFO, break_line=True, print_header=True, stack_displacement=1,
                    raise_error=True, adaptive_width=True):
        if log_level < self.log_level:
            return -1

        if self.dir_logs and not self.file_txt:
            raise Exception('Critical: Log file not defined. Do you have write permissions for {}?'.format(self.dir_logs))

        caller_info = inspect.getframeinfo(inspect.stack()[stack_displacement][0])

        message = ' '.join([str(m) for m in list(message)])

        if print_header:
            message_header = '[{} {:%Y-%m-%d %H:%M:%S}]'.format(self.indicator[log_level],
                                                                datetime.datetime.now())
            filename = caller_info.filename
            if adaptive_width:
                # allows the lineno_width to grow when necessary
                lineno_width = len(str(caller_info.lineno))
                self.max_lineno_width = max(lineno_width, self.max_lineno_width)
            else:
                # manually fix it to 3 numbers
                lineno_width = 3

            if len(filename) > 28 - self.max_lineno_width:
                filename = '...{}'.format(filename[-22 - (self.max_lineno_width - lineno_width):])

            message_locate = '{}.{}:'.format(filename, caller_info.lineno)
            message_logger = '{} {} {}'.format(message_header, message_locate, message)
            message_screen = '{}{}{}{} {} {}'.format(self.Colors.BOLD,
                                                     self.colorcode[log_level],
                                                     message_header,
                                                     self.Colors.END,
                                                     message_locate,
                                                     message)
        else:
            message_logger = message
            message_screen = message

        if break_line:
            print(message_screen)
            if self.dir_logs:
                self.file_txt.write('%s\n' % message_logger)
        else:
            print(message_screen, end='')
            sys.stdout.flush()
            if self.dir_logs:
                self.file_txt.write(message_logger)

        if self.dir_logs:
            self.file_txt.flush()
        if log_level == self.ERROR and raise_error:
            raise Exception(message)

    def log_dict_message(self, group, dictionary, description='', stack_displacement=2, log_level=SUMMARY):
        if log_level < self.log_level:
            return -1

        def print_subitem(prefix, subdictionary, stack_displacement=3):
            for key, value in sorted(subdictionary.items()):
                message = prefix + key + ':'
                if not isinstance(value, collections.Mapping):
                    message += ' ' + str(value)
                self.log_message(message, log_level=log_level, stack_displacement=stack_displacement)
                if isinstance(value, collections.Mapping):
                    print_subitem(prefix + '  ', value, stack_displacement=stack_displacement + 1)

        self.log_message('{}: {}'.format(group, description), log_level=log_level, stack_displacement=stack_displacement)
        print_subitem('  ', dictionary, stack_displacement=stack_displacement + 1)

    def _execute(self, statement, parameters=None, commit=True):
        parameters = parameters or ()
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        return_value = self.sqlite_cur.execute(statement, parameters)
        if commit:
            self.sqlite_conn.commit()
        return return_value

    def _run_query(self, query, parameters=None):
        return self._execute(query, parameters, commit=False)

    def _get_internal_table_name(self, table_name):
        return f'_{table_name}'

    def _check_table_exists(self, table_name):
        table_name = self._get_internal_table_name(table_name)
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        return self._run_query(query, table_name)

    def _create_table(self, table_name):
        table_name = self._get_internal_table_name(table_name)
        statement = f"""
            CREATE TABLE {table_name} (
                "__id" INTEGER PRIMARY KEY AUTOINCREMENT, -- rowid
                "__timestamp" DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        self._execute(statement, ())

    def _list_columns(self, table_name):
        table_name = self._get_internal_table_name(table_name)
        query = "SELECT name FROM PRAGMA_TABLE_INFO(?)"
        qry_cur = self._run_query(query, (table_name,))
        columns = (res[0] for res in qry_cur)
        # remove __id and __timestamp columns
        columns = [c for c in columns if not c.startswith('__')]
        return columns

    @staticmethod
    def _get_data_type(value):
        if isinstance(value, str):
            return 'TEXT'
        if isinstance(value, numbers.Number):
            return 'NUMERIC'
        raise ValueError('Only text and numeric are supported for now')

    def _add_column(self, table_name, column_name, value_sample=None):
        table_name = self._get_internal_table_name(table_name)
        value_type = self._get_data_type(value_sample)
        statement = f'ALTER TABLE {table_name} ADD COLUMN "{column_name}" {value_type}'
        return self._execute(statement)

    def _flatten_dict(self, dictionary, flatten_dict=None, prefix=''):
        flatten_dict = flatten_dict or {}
        for key, value in dictionary.items():
            local_prefix = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                self._flatten_dict(value, flatten_dict, prefix=local_prefix)
            elif not isinstance(value, (float, int)):
                raise TypeError(f'Invalid value type {type(value)} for {local_prefix}')
            else:
                flatten_dict[local_prefix] = value
        return flatten_dict

    def _insert_row(self, table_name, flat_dictionary):
        columns = [f'"{c}"' for c in self._list_columns(table_name)]
        table_name = self._get_internal_table_name(table_name)
        column_string = ', '.join(columns)
        value_placeholder = ', '.join(['?'] * len(columns))
        statement = f'INSERT INTO {table_name} ({column_string}) VALUES({value_placeholder})'
        parameters = tuple(val for val in flat_dictionary.values())
        return self._execute(statement, parameters)

    def log_dict(self, group, dictionary, description='', stack_displacement=2, should_print=False, log_level=SUMMARY):
        if log_level < self.log_level:
            return -1

        flat_dictionary = self._flatten_dict(dictionary)
        if self._check_table_exists(group).fetchone():
            columns = self._list_columns(group)
            for key in flat_dictionary:
                if key not in columns:
                    self.log_message('Key "{}" is unknown. New keys are not allowed'.format(key), log_level=self.ERROR)
            for column_name in columns:
                if column_name not in flat_dictionary:
                    self.log_message('Key "{}" not in the dictionary to be logged'.format(column_name), log_level=self.ERROR)
        else:
            self._create_table(group)
            for key, value in flat_dictionary.items():
                self._add_column(group, key, value)

        self._insert_row(group, flat_dictionary)

        if should_print:
            self.log_dict_message(group, dictionary, description, stack_displacement + 1, log_level)

    def init_sqlite(self):
        pre_existing = os.path.isfile(self.sqlite_file)
        self.sqlite_conn = sqlite3.connect(self.sqlite_file)
        self.sqlite_cur = self.sqlite_conn.cursor()
        if not pre_existing:
            self._create_table('bootstrap')

    def flush(self):
        if self.dir_logs:
            self.sqlite_conn.commit()
