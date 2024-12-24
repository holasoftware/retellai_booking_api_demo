import json
import logging
from dataclasses import dataclass


from litellm import acompletion

from .exceptions import ValidationError

logger = logging.getLogger(__name__)


class ReadonlyDict:
    def __init__(self, _dict):
        self._dict = _dict

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()


class ComputedValues:
    def __init__(self, compute_functions, data):
        self._compute_functions = compute_functions
        self._cached_precomputed_values = {}
        self._data = data

    def __getitem__(self, key):
        if key in self._compute_functions:
            if key not in self._cached_precomputed_values:
                compute_function = self._compute_functions[key]
                self._cached_precomputed_values[key] = compute_function(self._data)

            return self._cached_precomputed_values[key]
        else:
            raise KeyError(key)

    def __contains__(self, key):
        return key in self._compute_functions

    def __len__(self):
        return len(self._compute_functions)


class TransitionFuncWithConditions:
    def __init__(self, conditions=None):
        self.conditions = conditions

    def check_conditions(self, data):
        for condition in self.conditions:
            if condition['condition_function'](data):
                return condition['next_state']
        return None

    def add_condition(self, next_state, condition_function):
        self.conditions.append({
            "next_state": next_state,
            "condition_function": condition_function
        })

    def __call__(self, data):
        return self.check_conditions(data)


class LLMFSMState:
    def __init__(self, state_key, system_message=None, user_input=None, chat_history=None, tools=None, output_var="result", llm_model=None, temperature=None, function_def_transition_selector=None, tool_prefix_varname=None, output_parser=None, validate_json_response=None, response_format=None, chat_completion_extra_kwargs=None, tools_key="tools", precomputed_values=None, data=None):
        """
        - model (str): The LLM model to use for generating responses (default: "gpt-4o").
        """
        self.state_key = state_key
        self._system_message = system_message
        self._user_input = user_input
        self._chat_history = chat_history

        self.output_var = output_var
        self.llm_model = llm_model
        self.temperature = temperature
        self.function_def_transition_selector = function_def_transition_selector
        self.data = data
        self._readonly_data = ReadonlyDict(self.data)
        self.tool_prefix_varname = tool_prefix_varname
        self.output_parser = output_parser
        self.validate_json_response = validate_json_response
        self.tools_key = tools_key
        self.tools = tools
        self.precomputed_values = precomputed_values

        self.chat_completion_extra_kwargs = chat_completion_extra_kwargs
        self.response_format = response_format
 
        # TODO: check if llm_model accepts specific response_format
    def init_precomputed_values(self):
        if self.precomputed_values is not None:
            data = self.data
            data["precomputed_values"] = ComputedValues(self.precomputed_values, data)

    @property
    def is_bound(self):
        return self.data is not None

    def get_prompt_system_message(self):
        if self._system_message:
            if callable(self._system_message):
                return self._system_message(self._readonly_data)
            else:
                return self._system_message.format(self.data)

    def get_prompt_user_input(self):
        if self._user_input:
            if callable(self._user_input):
                return self._user_input(self._readonly_data)
            else:
                return self._user_input.format(self.data)

    def get_prompt_chat_history(self):
        if self._chat_history is None:
            return []
        else:
            chat_history = []
            for message in self._chat_history:
                if message["role"] == "placeholder":
                    chat_history.extend(message["content"])
                else:
                    content_template = message["content"]
                    if callable(content_template):
                        content = content_template(self._readonly_data) 
                    else:
                        content = content_template.format(self.data) 

                    chat_history.append({
                        "role": message["role"],
                        "content": content
                    })

            return chat_history

    def get_messages(self):
        messages = self.get_prompt_chat_history()
        if messages is None:
            messages = []
        else:
            messages = list(messages)

        system_message = self.get_prompt_system_message()
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        user_input = self.get_prompt_user_input()
        if user_input:
            messages.append({"role": "user", "content": user_input})

        return messages

    def get_chat_completion_extra_kwargs(self):
        if self._chat_completion_extra_kwargs:
            return self._chat_completion_extra_kwargs
        else:
            return {}

    async def step(self):
        self.init_precomputed_values()

        messages = self.get_messages()
        tools = self.tools

        kw = {
            "model": self.llm_model,
            "messages": messages
        }

        if self.tools is not None:
            kw["tools"] = self.tools

        if self.temperature is not None:
            kw["temperature"] = self.temperature

        if self.response_format is not None:
            kw["response_format"] = self.response_format

        if self.chat_completion_extra_kwargs is not None:
            kw.update(self.chat_completion_extra_kwargs)

        response = await acompletion(**kw)

        logger.info(
            f"tokens: {response.usage.total_tokens} total; {response.usage.completion_tokens} completion; {response.usage.prompt_tokens} prompt"
        )

        message = response.choices[0].message
        self.update_data(message)

        next_state_key = self.function_def_transition_selector(self._readonly_data)
        return next_state_key

    def process_assistant_message_content(self, assistant_answer):
        response_format = self.response_format

        if response_format is not None:
            if response_format["type"] == "json":
                result = json.loads(assistant_answer)
                if self.validate_json_response:
                    result = self.validate_json_response(result)
            elif response_format["type"] == "json_object":
                result = json.loads(assistant_answer)
        else:
            if self.output_parser:
                result = self.output_parser(assistant_answer)
            else:
                result = assistant_answer

        if isinstance(result, str):
            output_var = self.output_var or "result"

            assistant_output= {
                output_var: result
            }
        else:
            if self.output_var:
                assistant_output = {
                    self.output_var: result
                }
            else:
                assistant_output = result

        return assistant_output

    def update_data(self, message):
        data = self.data

        assistant_answer = message.content

        try:
            assistant_output = self.process_assistant_message_content(assistant_answer)
        except ValidationError as f:
            logger.warn(f"Validation error processing assistant answer: {f}")

        data.update(assistant_output)

        tool_calls = message.tool_calls
        if tool_calls:
            tool_calls_data = {}
            for tool_call in tool_calls:
                tool_calls_data[tool_call.function.name] = json.loads(tool_call.function.arguments)

            tool_prefix_varname = self.tool_prefix_varname

            if tool_prefix_varname:
                for tool_name, tool_args in tool_calls_data.items():
                    for variable_name, variable_value in new_vars.items():
                        data[tool_prefix_varname + variable_name] = variable_value
            else:
                data.update(tool_calls_data)

            data[self.tools_key] = tool_calls_data

    def get_clone_kwargs(self):
        return {
            "state_key": self.state_key,
            "system_message": self._system_message,
            "user_input": self._user_input,
            "chat_history": self._chat_history,
            "output_var": self.output_var,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "function_def_transition_selector": self.function_def_transition_selector,
            "data": self.data,
            "tool_prefix_varname": self.tool_prefix_varname,
            "output_parser": self.output_parser,
            "validate_json_response": self.validate_json_response,
            "tools": self.tools,
            "tools_key": self.tools_key,
            "precomputed_values": self.precomputed_values,
            "chat_completion_extra_kwargs": self.chat_completion_extra_kwargs,
            "response_format": self.response_format
        }

    def clone(self, **updated_kwargs):
        kwargs = self.get_clone_kwargs()

        if updated_kwargs:
            kwargs.update(updated_kwargs)

        return self.__class__(**kwargs)

    async def __call__(self, data):
        state = self.clone(data=data)
        result = await state.step()
        return result


@dataclass
class ManualResponse:
    user_intent: str
    answer: str


class ConversationFSMState(LLMFSMState):
    def __init__(self, user_input_key="user_input", assistant_answer_key="assistant_answer",  chat_history_key="chat_history", preprocess_input=None, restart_chat_history=False, goal=None, responses_per_user_intent=None, out_of_scope=None, information_to_be_gathered=None, confirmation=None, complete_string=None, **kwargs):
        super().__init__(**kwargs)
        self.user_input_key = user_input_key
        self.assistant_answer_key = assistant_answer_key
        self.chat_history_key = chat_history_key
        self.restart_chat_history = restart_chat_history
        self._preprocess_input = preprocess_input
        self.goal = goal
        self.responses_per_user_intent = responses_per_user_intent
        self.out_of_scope = out_of_scope
        self.information_to_be_gathered = information_to_be_gathered
        self.confirmation = confirmation
        self.complete_string = complete_string

    def preprocess_input(self, user_input):
        if self._preprocess_input is None:
            return user_input
        else:
            return self._preprocess_input(user_input)
    
    def process_assistant_message_content(self, assistant_answer):
        assistant_output = super().process_assistant_message_content(assistant_answer)
        assistant_output[self.assistant_answer_key] = assistant_answer

        return assistant_output

    def append_chat_history_message(self, role, content):
        if self.chat_history_key is None:
            self.data[self.chat_history_key]= {
                "role": role,
                "content": content
            }

        else:
            self.data[self.chat_history_key].append({
                "role": role,
                "content": content
            })

    def update_data(self, message):
        super().update_data(message)
        self.update_chat_history_data()

    def update_chat_history_data(self):
        self.append_chat_history_message("user", self.data[self.user_input_key])
        self.append_chat_history_message("assistant", self.data[self.assistant_answer_key])

    def get_prompt_system_message(self):
        system_message = super().get_prompt_system_message()
        if system_message:
            return system_message

        system_message = ""

        if self.goal:
            if callable(self.goal):
                goal = self.goal(self._readonly_data)
            else:
                goal = self.goal.format(self.data)

            system_message = "Your goal is: {goal}\n\n".format(goal=goal)

        if self.information_to_be_gathered:
            comma_separed_fields = ', '.join(self.information_to_be_gathered)
            system_message += "Information to be gathered: {comma_separed_fields}. This is all of the information you are to gather from the user, do not ask for anything else.".format(comma_separed_fields=comma_separed_fields)

            if self.confirmation:
                system_message += "Once you have the information ask for a confirmation."

                if self.complete_string:
                    system_message += "If you receive this confirmation reply only with:\n {completed_string}".format(completed_string=self.completed_string)

            elif self.complete_string:
                system_message += "Once you have the information reply only with:\n{completed_string}".format(completed_string=self.completed_string)

        if self.responses_per_user_intent:
            for manual_response in self.responses_per_user_intent:
                system_message += "\n\nIf the user wants {user_intent} reply only with this: {answer}".format(user_intent=manual_response['user_intent'], answer=manual_response['answer'])

            if self.out_of_scope:
                system_message += "\n\nFor any other user intention, answer: " + self.out_of_scope

        return system_message

    def get_prompt_user_input(self):
        user_input = super().get_prompt_user_input()

        if user_input is None:
            user_input = self.data[self.user_input_key]
            user_input = self.preprocess_input(user_input)
        return user_input

    def get_prompt_chat_history(self):
        if self.restart_chat_history:
            return []
        else:
            chat_history = self.data.get(self.chat_history_key)
            if chat_history:
                return chat_history
            else:
                return []

    def get_clone_kwargs(self):
        kwargs = super().get_clone_kwargs()
        kwargs["user_input_key"] = self.user_input_key
        kwargs["chat_history_key"] = self.chat_history_key
        kwargs["restart_chat_history"] = self.restart_chat_history
        kwargs["preprocess_input"] = self._preprocess_input
        kwargs["goal"] = self.goal
        kwargs["responses_per_user_intent"] = self.responses_per_user_intent
        kwargs["out_of_scope"] = self.out_of_scope
        kwargs["information_to_be_gathered"] = self.information_to_be_gathered
        kwargs["confirmation"] = self.confirmation
        kwargs["complete_string"] = self.complete_string

        return kwargs