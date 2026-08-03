import importlib
import pkgutil
import kt_lib.model.sequential_kt_model as kt_models
import kt_lib.model.cognitive_diagnosis_model as cd_models
import kt_lib.model.exercise_recommendation_model as er_models
import kt_lib.model.learning_path_recommendation_agent as lpr_agents


def import_all_models():
    for package in [kt_models, cd_models, er_models, lpr_agents]:
        package_dir = package.__path__[0]
        prefix = package.__name__ + "."
        for _, name, _ in pkgutil.iter_modules([package_dir]):
            importlib.import_module(prefix + name)

