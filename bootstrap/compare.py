import argparse
from os import path as osp
from tabulate import tabulate
import sqlite3
from contextlib import closing

# def get_internal_table_name(table_name):
#     return f'_{table_name}'

# def run_query(conn, query, parameters=None, cursor=None):
#     return execute(conn, query, parameters, commit=False, cursor=cursor)

# def list_columns(conn, table_name):
#     table_name = get_internal_table_name(table_name)
#     query = "SELECT name FROM PRAGMA_TABLE_INFO(?)"
#     with closing(conn.cursor()) as cursor:
#         qry_cur = run_query(conn, query, (table_name,), cursor=cursor)
#         columns = (res[0] for res in qry_cur)
#         # remove __id and __timestamp columns
#         columns = [c for c in columns if not c.startswith('__')]
#     return columns

# def select(conn, group, columns=None, where=None):
#     table_name = get_internal_table_name(group)
#     table_columns = list_columns(conn, group)
#     if columns is None:
#         column_string = '*'
#     else:
#         for c in columns:
#             if c not in table_columns:
#                 Logger()(f'Unknown column "{c}"', log_level=Logger.ERROR)
#         column_string = ', '.join([f'"{c}"' for c in columns])
#     statement = f'SELECT {column_string} FROM {table_name}'
#     with closing(conn.cursor()) as cursor:
#         return execute(conn, statement, cursor=cursor, commit=False).fetchall()


def execute(conn, statement, parameters=None, commit=True, cursor=None):
    assert parameters is None or isinstance(parameters, tuple)
    parameters = parameters or ()
    return_value = cursor.execute(statement, parameters)
    if commit:
        conn.commit()
    return return_value


def load_table(list_dir, metric, nb_epochs=None, best=None):
    table = []
    for dir_logs in list_dir:
        # if metric['fname'] == best['fname']:
        #     path_sql = osp.join(dir_logs, f'{metric["fname"]}.sqlite')
        #     conn = sqlite3.connect(path_sql, check_same_thread=False, isolation_level='IMMEDIATE')
        #     statement = f'SELECT m.{metric["column"]}, m.epoch FROM _{metric["group"]} AS m, _{best["group"]} AS b'
        #     if nb_epochs:
        #         statement += f' WHERE m.epoch < {nb_epochs}'
        #     if best['order'] == 'max':
        #         order = 'DESC'
        #     elif best['order'] == 'min':
        #         order = 'ASC'
        #     statement += f' ORDER BY b.{best["column"]} {order} LIMIT 1'
        #     with closing(conn.cursor()) as cursor:
        #         score, epoch = execute(conn, statement, cursor=cursor).fetchone()
        # else:
        path_sql = osp.join(dir_logs, f'{best["fname"]}.sqlite')
        conn = sqlite3.connect(path_sql, check_same_thread=False, isolation_level='IMMEDIATE')
        statement = f'SELECT {best["column"]}, epoch FROM _{best["group"]}'
        if nb_epochs:
            statement += f' WHERE epoch < {nb_epochs}'
        if best['order'] == 'max':
            order = 'DESC'
        elif best['order'] == 'min':
            order = 'ASC'
        statement += f' ORDER BY {best["column"]} {order} LIMIT 1'
        with closing(conn.cursor()) as cursor:
            best_score, best_epoch = execute(conn, statement, cursor=cursor).fetchone()

        path_sql = osp.join(dir_logs, f'{metric["fname"]}.sqlite')
        conn = sqlite3.connect(path_sql, check_same_thread=False, isolation_level='IMMEDIATE')
        statement = f'SELECT {metric["column"]}, epoch FROM _{metric["group"]}'
        statement += f' WHERE epoch == {best_epoch}'
        with closing(conn.cursor()) as cursor:
            score, epoch = execute(conn, statement, cursor=cursor).fetchone()

        table.append([dir_logs, score, epoch])

    if best['order'] == 'max':
        reverse = True
    elif best['order'] == 'min':
        reverse = False
    table.sort(key=lambda x: x[1], reverse=reverse)

    for i, x in enumerate(table):
        x.insert(0, f'# {i+1}')
    return table


def metric_str_to_dict(metric):
    split_ = metric.split('.')
    return {
        'fname': split_[0],
        'group': split_[1],
        'column': split_[2],
        'order': split_[3]
    }


def display_metrics(list_dir, metrics, nb_epochs=None, best=None):
    best = metric_str_to_dict(best)
    for mstr in metrics:
        metric = metric_str_to_dict(mstr)
        table = load_table(list_dir, metric, nb_epochs=nb_epochs, best=best)
        print(f'\n\n## {mstr}\n')
        print(tabulate(table, headers=['Place', 'Method', 'Score', 'Epoch']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--nb_epochs', default=-1, type=int)
    parser.add_argument('-d', '--dir_logs', default='', type=str, nargs='*')
    parser.add_argument('-m', '--metrics', type=str, nargs='*',
                        default=['logs.eval_epoch.accuracy.max',
                                 'logs.train_epoch.loss.min',
                                 'logs.train_epoch.accuracy.max'])
    parser.add_argument('-b', '--best', type=str,
                        default='logs.eval_epoch.accuracy.max')
    args = parser.parse_args()
    nb_epochs = None if args.nb_epochs == -1 else args.nb_epochs
    display_metrics(args.dir_logs, args.metrics, nb_epochs, args.best)
