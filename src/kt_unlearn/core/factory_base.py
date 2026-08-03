from abc import ABCMeta
import re

from kt_unlearn.core.base import Base
from kt_unlearn.utils.logger import GLogger


def _safe_log_text(value):
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


class ConfigurableFactory(Base, metaclass=ABCMeta):
    def __init__(self, global_ctx):
        super().__init__(global_ctx)

    def get_object(self, local_ctx):
        base_obj = get_class(local_ctx.config["class"])(self.global_ctx, local_ctx)
        self.info("Created Configurable: " + _safe_log_text(local_ctx.config["class"]))
        return base_obj


def get_instance_kvargs(kls, param):
    GLogger.getLogger().info("Instantiating: " + _safe_log_text(kls))
    return get_class(kls)(**param)


def get_function(func):
    func = get_class(func)
    GLogger.getLogger().info("Function: " + str(func))
    return func


def get_instance_config(config):
    GLogger.getLogger().info("Instantiating: " + _safe_log_text(config["class"]))
    return get_class(config["class"])(**config["parameters"])


def get_instance(kls, param):
    GLogger.getLogger().info("Instantiating: " + _safe_log_text(kls))
    return get_class(kls)(param)


def get_class(kls):
    parts = kls.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


__cls_param_ptrn = re.compile(r"(^.*)\((.*)\)")


def build_w_params_string(class_parameters):
    if isinstance(class_parameters, str):
        res = __cls_param_ptrn.findall(class_parameters)
        if len(res) == 0:
            return get_class(class_parameters)()
        return get_class(res[0][0])(**eval(res[0][1]))
    return class_parameters
