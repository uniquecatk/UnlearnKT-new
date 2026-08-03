import sys
import traceback
import torch
import gc
from kt_unlearn.core.base import Configurable
from kt_unlearn.core.factory_base import get_instance_kvargs
from kt_unlearn.evaluations.evaluation import Evaluation
from kt_unlearn.evaluations.running import UnlearnRunner
from kt_unlearn.utils.config.global_ctx import Global
from kt_unlearn.core.unlearner import Unlearner
from kt_unlearn.utils.config.local_ctx import Local


class Evaluator(Configurable):

    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.__init_measures__()

    def evaluate(self, unlearner: Unlearner, predictor):
        e = Evaluation(unlearner,predictor)
        for measure in self.measures:
            try:
                e = measure.process(e)
            except Exception as err:
                self.global_ctx.logger.warning(f"Error occurred during execution of evaluation {measure}")
                self.global_ctx.logger.warning(repr(err))
                if isinstance(measure, UnlearnRunner):
                    traceback.print_exc()

        self.global_ctx.logger.warning(f"Cleaning evaluator cache...")

        del e
        del unlearner
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
        
        return None

    def __init_measures__(self):
        self.measures = []
        for measure in self.params['measures']:
            current = Local(measure)
            self.measures.append( self.global_ctx.factory.get_object(current) )

        # the first metric has to be one that calls the unlearn() method of the unlearner
        if not isinstance(self.measures[0], UnlearnRunner):
            config = {"class": "kt_unlearn.evaluations.running.UnlearnRunner", "parameters":{}}
            current = Local(config)
            self.measures.insert(0, self.global_ctx.factory.get_object(current))

        assert isinstance(self.measures[0], UnlearnRunner)

