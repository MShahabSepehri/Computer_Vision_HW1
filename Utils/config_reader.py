import configparser
import os


def param_config(path):
    parent_dir = os.getcwd()
    config = configparser.ConfigParser()
    config.read(parent_dir + '/Config/' + path)
    params_parser = config['params']
    params = {}
    for key in params_parser:
        try:
            # noinspection PyTypeChecker
            params[key] = float(params_parser[key])
        except:
            params[key] = params_parser[key]
    return params
