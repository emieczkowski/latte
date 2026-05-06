from abc import ABC, abstractmethod

class AgentInterface(ABC):
    """
    Insert spec here.
    """

    @abstractmethod
    def __init__(self, name, system_prompt, model):
        """
        Initialize the agent with required fields.

        Args:
            name (str): The agent's name
            system_prompt (str): The system prompt that defines the agent's behavior
            model (dict, optional): Model configuration for LLM calls
        """
        pass

    @abstractmethod
    def receive(self, message, sender):
        """
        Receive an incoming message and add it to the buffer.

        Args:
            message (str): The message content
            sender (str, optional): The name of the message sender
        """
        pass

    @abstractmethod
    def reply(self):
        """
        Reply to all messages in buffer and move them to history.

        Args:
            verbose (bool): If True, enable verbose output for LLM calls

        Returns:
            str: The agent's reply message
        """
        pass