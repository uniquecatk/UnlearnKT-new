from abc import ABCMeta, abstractmethod

from kt_unlearn.core.base import Configurable
from kt_unlearn.evaluations.manager import Evaluation


class Measure(Configurable, metaclass=ABCMeta):

    @abstractmethod
    def process(self, e:Evaluation):
        return e
