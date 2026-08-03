from kt_unlearn.data.preprocessing.preprocess import Preprocess
from kt_unlearn.utils.config.global_ctx import Global
from kt_unlearn.utils.config.local_ctx import Local

class BBBP_preprocess(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)

    def process(self, X, y, z):

        return X,y,z