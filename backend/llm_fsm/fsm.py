import logging
from functools import wraps
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type, List


from litellm.types.completion import ChatCompletionMessageParam


from .fsm_state import (
    LLMFSMState, ConversationFSMState
)
from .exceptions import FSMError, TransitionException, TransitionRequired, TransitionsNotAllowed, InvalidTransition

logger = logging.getLogger(__name__)


@dataclass
class FSMRun:
    i: int
    state: str
    context_data: dict


END_STATE = "end"
START_STATE = "start"

class LLMStateMachine:
    """
    Finite State Machine for LLM-driven Agents.
    This class enables the creation of conversational agents by defining states, transitions using structured responses from LLMs.

    Parameters:
    - _state: The current active state of the FSM, representing the ongoing context of the conversation.
    - _initial_state: The starting state of the FSM, used for initialization and resetting.
    - _end_state: The terminal state of the FSM, where processing ends (default is "END").
    - _state_registry: A dictionary that stores metadata about all defined states and their transitions.
    - data: A dictionary for storing the context data..
    """
    llm_state_class = LLMFSMState

    def __init__(self, initial_state: str = START_STATE, end_state: str = END_STATE, allowed_transitions: Dict[str, str] = None, default_llm_model: str | None = None, default_temperature: str | None = None, common_tools=None):
        self._state = initial_state
        self._initial_state = initial_state
        self._end_state = end_state
        self._state_registry = {}
        self._state_transition_log = []
        self._allowed_transitions_per_node = allowed_transitions or {}
        self._started = False
        self._default_llm_model = default_llm_model
        self._default_temperature = default_temperature
        self._common_tools = common_tools

        self.data = {}

    @property
    def current_state(self):
        return self._state

    @property
    def current_state_node(self):
        return self._state_registry[self._state]

    def add_state_transition(self, start_state, end_state):
        if start_state not in self._state_registry:
            self._allowed_transitions_per_node[start_state] = []

        self._allowed_transitions_per_node[start_state].append(end_state)

    @property    
    def started(self):
        return self._started

    def on_complete(self, data):
        # Default behavior: proceed to the next action or return data
        return data

    def add_state_callback(self, state_key, func):
        self._state_registry[state_key] = func

    def define_state(
        self,
        function_def_transition_selector: Callable| None=None,
        state_key: str | None =None,
        temperature: float = 0.5,
        llm_model: str | None =None,
        system_message: str | None = None,
        user_input: str | None = None,
        chat_history: list | None = None,
        tools: list | Callable | None = None,
        **kwargs
    ):
        """
        Decorator to define and register a state [@fsm.define_state(...)] in the FSM (Finite State Machine).

        This function simplifies the process of associating metadata (such as prompts and transitions) 
        with a Python function that defines the behavior of the state.

        Parameters:
        - state_key (str): A unique identifier for the state.
        - system_message (str): Instructions provided to the LLM when this state is active.
        - preprocess_prompt_template (Optional[Callable]): A func to preprocess the sys prompt for the LLM.
        - temperature (float): Determines the randomness of LLM responses, defaults at 0.5.
        Returns:
        - callable The original function wrapped and registered with the FSM.
        """

        if llm_model is None:
            llm_model = self._default_llm_model

        if temperature is None:
            temperature = self._default_temperature

        if self._common_tools:
            if tools is None:
                tools = self._common_tools
            else:
                tools = self._common_tools + tools

        if function_def_transition_selector is None:               
            def decorator(func: Callable):
                nonlocal state_key
                if state_key is None:
                    state_key = func.__name__

                @wraps(func)
                async def wrapper(*args, **kwargs):
                    return await func(*args, **kwargs)

                # Register the state in the FSM's registry with the provided metadata
                self.add_state_callback(state_key, self.llm_state_class(
                    state_key=state_key,
                    temperature=temperature,
                    llm_model=llm_model,
                    function_def_transition_selector=wrapper,
                    system_message=system_message,
                    user_input=user_input,
                    chat_history=chat_history,
                    tools=tools,
                    **kwargs
                ))
                return wrapper
            return decorator
        else:
            if state_key is None:
                state_key = function_def_transition_selector.__name__

            @wraps(function_def_transition_selector)
            async def wrapper(*args, **kwargs):
                return await function_def_transition_selector(*args, **kwargs)

            self.add_state_callback(state_key, self.llm_state_class(
                state_key=state_key,
                temperature=temperature,
                llm_model=llm_model,
                function_def_transition_selector=wrapper,
                system_message=system_message,
                user_input=user_input,
                chat_history=chat_history,
                tools=tools,
                **kwargs
            ))

    async def run_state_machine(
        self,
        max_n=float("inf"),
        stop_before_state=None,
        data=None
    ) -> FSMRun:
        """
        Executes a single step of the Finite State Machine (FSM) using the provided user input.
        Processes the current state, generates a response via the LLM, and transitions the FSM.

        Parameters:
        - user_input (str): User input for the FSM.

        Returns:
        - FSMRun: A structured representation of the FSM's state, chat history, and response.
        """

        if self._started:
            self._started = False

        if data is not None:
            self.data.update(data)

        i = 0
        while i < max_n:
            state = self._state
            if stop_before_state == state:
                break

            state_node_func: LLMFSMState = self._state_registry.get(state)
            if not state_node_func:
                raise FSMError(f"State '{state}' not found in the state registry.")

            # Extract response and next state
            next_state = await state_node_func(self.data)

#            allowed_transitions = self._allowed_transitions_per_node.get(state)
#
#            if next_state is None:
#                if allowed_transitions:
#                    num_allowed_transitions = len(allowed_transitions)
#                else:
#                    num_allowed_transitions = 0
#
#                if num_allowed_transitions == 0:
#                    next_state = self._end_state
#                elif num_allowed_transitions == 1:
#                    next_state = allowed_transitions[0]
#                else:
#                    raise TransitionRequired()
#            else:
#                if allowed_transitions is None:
#                    raise TransitionsNotAllowed()
#                elif next_state not in allowed_transitions:
#                    raise InvalidTransition(f"Transition to {next_state} not allowed")
#
            if next_state is None:
                break

            self._state = next_state
            self._state_transition_log.append([state, next_state])

            if self.is_completed():
                self.on_complete()
                break

            if stop_before_state == next_state:
                break

            i += 1

        return FSMRun(
            i=i,
            state=self._state,
            context_data=self.data,
        )

    async def run_state_machine_until(
        self,
        *args,
        **kwargs,
    ) -> FSMRun:
        pass

    def reset(self):
        """Resets the FSM to its initial state."""
        self._state = self._initial_state
        self._state_transition_log = []
        self._started = False
        self.data = {}

    def set_context_data(self, key: str, value: Any):
        """Sets a key-value pair into the user-defined context."""
        self.data[key] = value

    def set_context_data_dict(self, data: Dict[str, Any]):
        """Sets multiple key-value pairs into the user-defined context."""
        self.data.update(data)

    def get_context_data(self, key: str, default: Any = None):
        """Gets a value from the user-defined context, with a default value."""
        return self.data.get(key, default)

    def get_full_context_data(self):
        """Returns the full user-defined context."""
        return self.data

    def is_completed(self):
        """Checks if the FSM has reached its final state."""
        return self._state == self._end_state


class ConversationalLLMStateMachine(LLMStateMachine):
    chat_history_key = "chat_history"
    user_input_key = "user_input"
    assistant_answer_key="assistant_answer"

    llm_state_class = ConversationFSMState

    def define_state(self, *args, **kwargs):
        return super().define_state(
            *args,
            chat_history_key=self.chat_history_key,
            user_input_key=self.user_input_key,
            assistant_answer_key=self.assistant_answer_key,
            **kwargs)

    def add_message(self, role, content):
        self.chat_history.append({"role": role, "content": content})

    @property
    def chat_history(self) -> List[ChatCompletionMessageParam]:
        return self.data[self.chat_history_key]

    @property
    def last_message(self) -> ChatCompletionMessageParam | None:
        chat_history = self.chat_history
        if len(chat_history) < 1:
            return None
        return chat_history[-1]

    async def run_state_machine(
        self,
        user_input: str,
        **kwargs,
    ) -> FSMRun:

        kwargs.setdefault("max_n", 1)

        self.data[self.user_input_key] = user_input
        result = await super().run_state_machine(**kwargs)

        return result

    async def ask(self, user_input):
        result = await self.run_state_machine(user_input)
        return result.state[self.assistant_answer_key]