import os
import yaml
import logging
import logging.config

__all__ = [
    "load_config",
    "setup_logging",
]

config_dir = lambda file: os.path.join(rtx.CONFIG_DIR, file)
logconfig_path = config_dir("logging.yaml")
default_level = logging.DEBUG

def load_config(path):
    """
    Helper function which reads the yaml config file and stores it 
    into json format.

    Args:
    -----
     path: str
      The path to the config file. 

    Returns:
    --------
     config: json file 
    """
    with open(path, 'r') as config_file:
        config = yaml.safe_load(config_file.read())
    return config

def setup_logging():
    """
    Initialises the logger with the configuration specified in the config/logging.yaml
    """
    logger = logging.getLogger()
    if os.path.exists(logconfig_path):
        config = load_config(logconfig_path)
        temp = config["handlers"]
        for key in temp.keys():  # this ensures that logs are stored in the log folder
            if 'filename' in temp[key]:
                temp[key]['filename'] = os.path.join(rtx.LOG_DIR, temp[key]['filename'])
        config["handlers"] = temp
        logging.config.dictConfig(config)
    else:
        logging.config.basicConfig(level=default_level)

    return logger

