#################################################################################
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE #
# SOFTWARE.                                                                     #
#################################################################################
 
import os
import sys
import json
import inspect
import datetime
import collections

# https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
class Logger(object):
    """ The Logger class is a singleton. It contains all the utilities
        for logging variables in a key-value dictionary.
        It can also be considered as a replacement for the print function.

        .. code-block:: python

            Logger(dir_logs='logs/mnist')
            Logger().log_value('train_epoch.epoch', epoch)
            Logger().log_value('train_epoch.mean_acctop1', mean_acctop1)
            Logger().flush() # write the logs.json

            Logger()("Launching training procedures") # written to logs.txt
            > [I 2018-07-23 18:58:31] ...trap/engines/engine.py.80: Launching training procedures
    """

    # Attributs

    INFO = 0
    SUMMARY = 1
    WARNING = 2
    ERROR = 3
    SYSTEM = 4
    _instance = None
    indicator = {INFO: 'I', SUMMARY: 'S', WARNING: 'W', ERROR: 'E', SYSTEM: 'S'}

    class Colors:
        END = '\033[0m'
        BOLD = '\033[1m'
        ITALIC = '\033[3m'
        UNDERLINE = '\033[4m'
        GREY = '\033[90m'
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        PURPLE = '\033[95m'
        SKY = '\033[96m'
        WHITE = '\033[97m'
    colorcode = {INFO: Colors.GREY, SUMMARY: Colors.BLUE, WARNING: Colors.YELLOW, ERROR: Colors.RED, SYSTEM: Colors.WHITE}

    compactjson = True
    log_level = None # log level
    dir_logs = None
    path_json = None
    path_txt = None
    file_txt = None
    name = None
    perf_memory = {}
    values = {}

    # Methods

    def __new__(self, dir_logs=None, name='logs'):
        if Logger._instance is None:

            Logger._instance = object.__new__(Logger)

            Logger._instance.set_level(Logger._instance.INFO)

            if dir_logs:
                Logger._instance.name = name
                Logger._instance.dir_logs = dir_logs
                Logger._instance.path_txt = os.path.join(dir_logs, '{}.txt'.format(name))
                Logger._instance.file_txt = open(os.path.join(dir_logs, '{}.txt'.format(name)), 'a+')
                Logger._instance.path_json = os.path.join(dir_logs, '{}.json'.format(name))
                Logger._instance.reload_json()
            else:
                Logger._instance.log_message('No logs files will be created (dir_logs attribut is empty)', log_level=Logger.WARNING)

        return Logger._instance


    def __call__(self, *args, **kwargs):
        return self.log_message(*args, **kwargs, stack_displacement=2)


    def set_level(self, log_level):
        self.log_level = log_level


    def set_json_compact(self, is_compact):
        self.compactjson = is_compact


    def log_message(self, *message, log_level=INFO, break_line=True, print_header=True, stack_displacement=1):
        """
        """
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
            if len(filename) > 25:
                filename = '...{}'.format(filename[-22:])

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
        if log_level==self.ERROR:
            raise Exception(message)


    def log_value(self, name, value, stack_displacement=2, should_print=False, log_level=SUMMARY):
        """
        """
        if name not in self.values:
            self.values[name] = []
        self.values[name].append(value)

        if should_print:
            if type(value) == float:
                if int(value) == 0:
                    message = '{}: {:.6f}'.format(name, value)
                else:
                    message = '{}: {:.2f}'.format(name, value)
            else:
                message = '{}: {}'.format(name, value)
            self.log_message(message, log_level=log_level, stack_displacement=stack_displacement+1, )


    def log_dict(self, group, dictionary, description='', stack_displacement=2, should_print=False, log_level=SUMMARY):
        """
        """
        if group not in self.perf_memory:
            self.perf_memory[group] = {}
        else:
            for key in self.perf_memory[group].keys():
                if key not in dictionary.keys():
                    self.log_message('Key "{}" not in the dictionary to be logged'.format(key), log_level=self.ERROR)
            for key in dictionary.keys():
                if key not in self.perf_memory[group].keys():
                    self.log_message('Key "{}" is unknown. New keys are not allowed'.format(key), log_level=self.ERROR)

        for key in dictionary.keys():
            if key in self.perf_memory[group]:
                self.perf_memory[group][key].extend([dictionary[key]])
            else:
                self.perf_memory[group][key] = [dictionary[key]]

        if should_print:
            self.log_dict_message(group, dictionary, description, stack_displacement+1, log_level)


    def log_dict_message(self, group, dictionary, description='', stack_displacement=2, log_level=SUMMARY):
        def print_subitem(prefix, subdictionary, stack_displacement=3):
            for key, value in sorted(subdictionary.items()):
                message = prefix + key + ':'
                if not isinstance(value, collections.Mapping):
                    message += ' ' + str(value)
                self.log_message(message, log_level=log_level, stack_displacement=stack_displacement)
                if isinstance(value, collections.Mapping):
                    print_subitem(prefix + '  ', value, stack_displacement=stack_displacement+1)

        self.log_message('{}: {}'.format(group, description), log_level=log_level, stack_displacement=stack_displacement)
        print_subitem('  ', dictionary, stack_displacement=stack_displacement+1)


    def reload_json(self):
        if os.path.isfile(self.path_json):
            try:
                with open(self.path_json, 'r') as json_file:
                    self.values = json.load(json_file)
            except:
                self.log_message('json log file can not be open: {}'.format(self.path_json), log_level=self.WARNING)


    def flush(self):
        """
        """
        if self.dir_logs:
            self.path_tmp = self.path_json + '.tmp'
            try:
                with open(self.path_tmp, 'w') as json_file:
                    if self.compactjson:
                        json.dump(self.values, json_file, separators=(',', ':'))
                    else:
                        json.dump(self.values, json_file, indent=4)
                if os.path.isfile(self.path_json):
                    os.system('rm '+self.path_json)
                os.system('mv {} {}'.format(self.path_tmp, self.path_json))
            except Exception as e:
                print(e)
