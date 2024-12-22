import json

from litellm import acompletion


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
    def __init__(self, state_key, system_message=None, user_input=None, chat_history=None, tools=None, keep_messages=False, output_var="result", llm_model=None, function_def_transition_selector=None, data=None, tool_prefix_varname="", process_agent_answer_content=None, process_tool_call=None, tools=None, response_format=None,  chat_completion_extra_kwargs=None):
        """
        - model (str): The LLM model to use for generating responses (default: "gpt-4o").
        """
        self.state_key = state_key
        self._system_message = system_message
        self._user_input = user_input
        self._chat_history = chat_history

        self.conditions = []
        self.keep_messages = keep_messages
        self.output_var = output_var
        self.llm_model = llm_model
        self.function_def_transition_selector = function_def_transition_selector
        self.data = data
        self.tool_prefix_varname = tool_prefix_varname
        self._process_agent_answer_content = process_agent_answer_content
        self._process_tool_call = process_tool_call
        self._tools = tools
        self._chat_completion_extra_kwargs = chat_completion_extra_kwargs
        self._response_format = response_format
 
    @property
    def is_bound(self):
        return self.data is not None

    def process_agent_answer_content(self, agent_answer):
        if self._process_agent_answer_content is not None:
            return self._process_agent_answer_content(agent_answer)
        else:
            return agent_answer

    def process_agent_answer_content(self, agent_answer):
        if self._process_agent_answer_content is not None:
            return self._process_agent_answer_content(agent_answer)
        else:
            return agent_answer

    def get_prompt_system_message(self):
        if self._system_message:
            return self._system_message.format(self.data)

    def get_prompt_user_input(self):
        if self._user_input:
            return self._user_input.format(self.data)

    def get_prompt_chat_history(self):
        if self._chat_history is None:
            return []
        else:
            chat_history = []
            for message in self._chat_history
                chat_history.append({
                    "role": message["role"],
                    "content": message["content"].format(self.data) 
                })

            return chat_history

    def get_tools(self):
        if self._tools:
            tools = self._tools
            if callable(tools):
                return tools(self.data)
            else:
                return tools

    def get_messages(self):
        messages = self.get_prompt_chat_history()

        system_message = self.get_prompt_system_message()
        if system_message is not None:
            messages.insert(0, {"role": "system", "content": system_message})

        user_input = self.get_prompt_user_input()
        if user_input is not None:
            messages.append({"role": "user", "content": user_input})

        return messages

    def get_chat_completion_extra_kwargs(self):
        if self._chat_completion_extra_kwargs:
            return self._chat_completion_extra_kwargs
        else:
            return {}

    async def step(self):
        messages = self.get_messages()
        tools = self.get_tools()

        response = await acompletion(
            model=self.llm_model,
            messages=messages,
            tools=tools,
            response_format=self._response_format,
            **self.get_chat_completion_extra_kwargs()
        )

        logger.info(
            f"tokens: {response.usage.total_tokens} total; {response.usage.completion_tokens} completion; {response.usage.prompt_tokens} prompt"
        )

        message = response.choices[0].message

        agent_answer = message.content

        if self._response_format is not None and self._response_format["type"] == "json" or self._response_format["type"] == "json_object":
            agent_answer = json.loads(agent_answer)

        tool_calls = message.tool_calls
        if tool_calls:
            tool_calls_with_args = {}
            for tool_call in tool_calls:
                tool_calls_with_args[tool_call.function.name] = json.loads(tool_call.function.arguments)
        else:
            tool_calls_with_args = None

        self.update_data(agent_answer=agent_answer, tool_calls_with_args=tool_calls_with_args)

        next_state_key = self.function_def_transition_selector(self.data)
        return next_state_key

    def update_data(self, agent_answer, tool_calls_with_args=None):
        data = self.data

        if self.response_model:
            try:
                parsed_response = self.response_model(**agent_answer)
            except ValidationError as error:
                raise FSMError(f"Error parsing response: {error}")
        else:
            parsed_response = raw_response.get("content", raw_response)

        agent_output = self.process_agent_answer_content(agent_answer)

        if isinstance(agent_answer, str):
            data[self.output_var] = agent_output
        else:
            data.update(agent_output)

        if tool_calls_with_args:
            tool_prefix_varname = self.tool_prefix_varname
            process_tool_call = self.process_tool_call

            for tool_name, tool_args in tool_calls_with_args.items():
                if process_tool_call:
                    new_vars = process_tool_call(tool_name, tool_args)
                else:
                    new_vars = tool_args

                for variable_name, variable_value in new_vars.items():
                    data[tool_prefix_varname + variable_name] = variable_value

    def get_clone_kwargs(self):
        return {
            "state_key": self.state_key,
            "prompt_template": self.prompt_template,
            "conditions": self.conditions,
            "keep_messages": self.keep_messages,
            "result_key": self.result_key
        }

    def clone(self, data):
        kwargs = self.get_clone_kwargs()
        kwargs["data"] = data
        return self.__class__(**kwargs)

    async def __call__(self, data):
        state = self.clone(data=data)
        return state.step()


class ConversationFSMState(LLMFSMState):

    def __init__(self, user_input_key="user_input", chat_history_key="chat_history", **kwargs):
        super().__init__(**kwargs)
        self.user_input_key = user_input_key
        self.chat_history_key = chat_history_key

    def preprocess_input(self, user_input):
        return user_input

    def get_chat_history(self):
        chat_history = list(self.data[self.chat_history_key])

        user_input = self.data[self.user_input_key]
        user_input = self.preprocess_input(user_input)

        if user_input is not None:
            chat_history.append({"role": "user", "content": user_input})

        return chat_history