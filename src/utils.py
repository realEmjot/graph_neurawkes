import abc


class VariablesContainer(abc.ABC):
    @abc.abstractmethod
    def get_variables_list(self):
        pass
