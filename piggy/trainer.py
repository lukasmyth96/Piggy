from abc import ABC, abstractmethod


class Trainer(ABC):

    def __init__(self, environment, agent):
        """
        Base class for all learning algorithms I'll implement
        Parameters
        ----------
        environment: piggy.environment.Environment
        agent: piggy.agent.Agent
        """
        self.environment = environment
        self.agent = agent

    @abstractmethod
    def train(self):
        """ This method is where the learning algorithm will be implemented"""
        raise NotImplementedError('You must implement a \'train\' method for every sub-class of Trainer')