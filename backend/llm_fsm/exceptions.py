class FSMError(Exception):
    pass

class TransitionException(FSMError):
    pass

class TransitionRequired(TransitionException):
    pass

class TransitionsNotAllowed(TransitionException):
    pass

class InvalidTransition(TransitionException):
    pass

class ValidationError(FSMError):
    pass

