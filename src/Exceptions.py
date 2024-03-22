class IllegalArgumentException(Exception):
    """
    Class that represents an illegal argument exception.

    Should be raised whenever some arguments differs from expected.
    """

    def __int__(self, message):
        """
        Constructor of the exception.

        Args:
            message (str): The message to display along with the exception.
        """
        self.message = message
        super().__init__(self.message)
