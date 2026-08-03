from copy import deepcopy
import json
import os
import re

from kt_unlearn.utils.config.file_parser import Config
from kt_unlearn.utils.logger import GLogger
import numpy as np
import torch
import random


GLOBAL_PLACEHOLDER_RE = re.compile(r"\$\{(?:globals\.)?([A-Za-z_][A-Za-z0-9_]*)\}")

class Global:

    logger = GLogger.getLogger()
    info = logger.info

    def __init__(self, config_file):

        self.info("Creating Global Context for: " + config_file)
        if not os.path.exists(config_file):
            raise ValueError(f'''The provided config file does not exist. PATH: {config_file}''')
        
        self.config = Config.from_json(config_file)
        self.__setglobals__()

    def __setglobals__(self):
        if not hasattr(self.config, 'globals'):
            self.config.globals={}

        self._configure_output_roots()
        self._expand_global_placeholders()

        if 'seed' in self.config.globals:
            self.info(f'''Setting seeds to: {self.config.globals['seed']}''' )
            self.set_seed(self.config.globals['seed'])
        else:
            gen_seed = random.SystemRandom().randint(0 , 2**32 - 1)
            self.config.globals['seed'] = gen_seed
            self.info(f'''{bcolors.FAIL}WARNING - SEEDS ARE RANDOMLY GENERATED AS {self.config.globals['seed']} - Add globals[\'seed\'] to the main Cfg to fix them.{bcolors.ENDC}''' )
            self.set_seed(gen_seed)

        if 'cached' not in self.config.globals:
            self.config.globals['cached'] = self.cached = False
        else:
            self.config.globals['cached'] = self.cached = strtobool(self.config.globals['cached'])
            
        self.info(f'''{bcolors.FAIL}Caching System: {self.cached}.{bcolors.ENDC}''' )

    def _configure_output_roots(self):
        results_root = os.getenv(
            "KT_RESULTS_ROOT",
            self.config.globals.get("results_root", "output/runs/erasure"),
        )
        split_root = os.getenv(
            "KT_SPLIT_ROOT",
            self.config.globals.get("split_root", "${results_root}/splits"),
        )
        self.config.globals["results_root"] = str(results_root).replace("\\", "/")
        self.config.globals["split_root"] = str(split_root).replace("\\", "/")

    def _expand_global_placeholders(self):
        globals_cfg = dict(self.config.globals)
        for _ in range(10):
            updated = {
                key: self._expand_value(value, globals_cfg)
                for key, value in globals_cfg.items()
            }
            if updated == globals_cfg:
                break
            globals_cfg = updated
        self.config.globals.update(globals_cfg)
        self.config.__dict__ = self._expand_value(self.config.__dict__, self.config.globals)
        self.info(
            "Resolved output roots: results_root=%s, split_root=%s",
            self.config.globals["results_root"],
            self.config.globals["split_root"],
        )

    def _expand_value(self, value, globals_cfg):
        if isinstance(value, dict):
            return {key: self._expand_value(val, globals_cfg) for key, val in value.items()}
        if isinstance(value, list):
            return [self._expand_value(item, globals_cfg) for item in value]
        if isinstance(value, str):
            current = value
            for _ in range(10):
                expanded = GLOBAL_PLACEHOLDER_RE.sub(
                    lambda match: str(globals_cfg.get(match.group(1), match.group(0))),
                    current,
                )
                if expanded == current:
                    break
                current = expanded
            return current
        return value



    def set_seed(self,seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        # For more deterministic behavior, you can set the following
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

  
def clean_cfg(cfg):
    if isinstance(cfg,dict):
        new_cfg = {}
        for k in cfg.keys():
            if hasattr(cfg[k],"local_config"):#k == 'oracle' or k == 'dataset':
                new_cfg[k] = clean_cfg(cfg[k].local_config)
            elif isinstance(cfg[k], (list,dict, np.ndarray)):
                new_cfg[k] = clean_cfg(cfg[k])
            else:
                new_cfg[k] = cfg[k]
    elif isinstance(cfg, (list, np.ndarray)):
        new_cfg = []
        for k in cfg:
            new_cfg.append(clean_cfg(k))
    else:
        new_cfg = cfg

    return new_cfg

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if not isinstance(val,bool):
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return True
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return False
        else:
            raise ValueError("invalid truth value %r" % (val,))
    else:
        return val
