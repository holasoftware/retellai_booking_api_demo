import os
import json
import logging
from typing import List


from openai import AsyncOpenAI

from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from .llm_fsm.fsm import LLMStateMachine


logger = logging.getLogger(__name__)


available_slots = {
    "2023-07-01": ["10:00", "11:00", "14:00", "15:00"],
    "2023-07-02": ["09:00", "10:00", "11:00", "14:00"],
    "2023-07-03": ["11:00", "13:00", "14:00", "16:00"],
    "2023-07-04": ["10:00", "12:00", "15:00", "16:00"],
}

appointments = []


def book_appointment(name, phone, email, appointment_date):
    # Implement your appointment booking logic here
    logger.info(f"Appointment booked for {name} on {appointment_date}. We will contact you at {phone} or {email}.")

    if date in available_slots[doctor_name] and time in available_slots[doctor_name][date]:
        available_slots[date].remove(time)
        appointments.append({
            "name": name,
            "phone": phone,
            "email": email,
            "appointment_date": appointment_date
        })
        return True
    return False


def check_availability(date):
    return available_slots.get(date, [])


begin_sentence = "Hey there, I'm your personal hair salon assistant, how can I help you?"
agent_prompt = """You are assisting customers with inquiries about our hair salon "Filpino haircuts". Please provide the following information so the customer can accurately answer their questions about services, pricing, and scheduling, ultimately improving customer satisfaction and efficiency:
 * List of all hair services offered: 
    - haircut
    - coloring
    - extensions
    - special hair treatment
 * Pricing for each service:
    - Women's Cut: 400 philippines pesos
    - Men's Cut: 300 philippines pesos
    - Full Highlights: 120-180 philippines pesos (depending on length and thickness)
    - Coloring: 400 philippines pesos
    - Make-up: 100 philippines pesos
 * Pricing for packages:
    - Wedding Package: Includes haircut, styling, and makeup for 1250 philippines pesos
 * Discounts
    - Student Discount: 10% off showing the student card
 * Salon hours of operation: 
    Monday-Friday: 9:00 AM - 7:00 PM
    Saturday: 9:00 AM - 5:00 PM
    Closed Sundays
 * Appointment policies: Appointments recommended, walk-ins welcome on availability. Online booking available through https:///www.hairsalon.ph
 * Contact information:
    Phone number: 0902392393
    Email: info@hairsalon.ph
    Salon address: SM Mega Mall, floor 1, beside SM supermarket
Optional additions:
 * Our team: 
    John Doe: Color Specialist
    Jane Smith: Cutting and Styling Expert
 * Information about the salon's atmosphere and amenities: Relaxing ambiance. Complimentary beverages, Free Wi-Fi.
 * Our hair salon "Filipino haircuts" provides exceptional hair and make-ups services with the highest level of customer satisfaction doing everything we can to meet your expectations
"""


class LlmClient:
    def __init__(self, model="gpt-4o-mini"):
        self.client = AsyncOpenAI(
            organization=os.environ.get("OPENAI_ORGANIZATION_ID"),
            api_key=os.environ["OPENAI_API_KEY"],
            model=model
        )
        self.model = model

    def draft_begin_message(self):
        response = ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )
        return response

    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({"role": "assistant", "content": utterance.content})
            else:
                messages.append({"role": "user", "content": utterance.content})
        return messages

    def prepare_prompt(self, request: ResponseRequiredRequest):
        prompt = [
            {
                "role": "system",
                "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n'
                + agent_prompt,
            }
        ]
        transcript_messages = self.convert_transcript_to_openai_messages(
            request.transcript
        )
        for message in transcript_messages:
            prompt.append(message)

        if request.interaction_type == "reminder_required":
            prompt.append(
                {
                    "role": "user",
                    "content": "(Now the user has not responded in a while, you would say:)",
                }
            )
        return prompt

    # Step 1: Prepare the function calling definition to the prompt
    def prepare_functions(self):
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "end_call",
                    "description": "End the call only when user explicitly requests it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message you will say before ending the call with the customer.",
                            },
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_user_intent",
                    "description": """According to the conversation, select one of the next user intents if it's possible:
    - appointment: the user wants to make an appointment
    - appointment_confirmation: the user confirms the information regarding the appointment
    - information_inquiry: the user wants information about the service
    - complain: the user is complaining
    - thanks: the user is complimenting the service
""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intention": {
                                "type": "string",
                                "description": "The intention of the user.",
                                "enum": ["appointment", "appointment_confirmation", "information_inquiry", "complain", "thanks"]
                            },
                        },
                        "required": ["intention"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_availability",
                    "description": "Check available time slots for a given date",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "The date to check availability for, in YYYY-MM-DD format"
                            }
                        },
                        "required": ["date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "schedule_appointment",
                    "description": "Schedule an appointment for the service if the user confirms the relevant information for the appointment: date, time, name, email and phone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "The date for the appointment, in YYYY-MM-DD format"
                            },
                            "time": {
                                "type": "string",
                                "description": "The time for the appointment, in HH:MM format"
                            },
                            "customer_name": {
                                "type": "string",
                                "description": "The name of the customer"
                            },
                            "customer_email": {
                                "type": "string",
                                "description": "The email of the customer"
                            },
                            "customer_phone": {
                                "type": "string",
                                "description": "The phone of the customer"
                            }
                        },
                        "required": ["date", "time", "customer_name", "customer_email", "customer_phone"]
                    }
                }
            }
        ]
        return functions

    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        func_call = {}
        func_arguments = ""
        stream = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Or use a 3.5 model for speed
            messages=prompt,
            stream=True,
            # Step 2: Add the function into your request
            tools=self.prepare_functions(),
        )

        async for chunk in stream:
            # Step 3: Extract the functions
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.tool_calls:
                tool_calls = chunk.choices[0].delta.tool_calls[0]
                if tool_calls.id:
                    if func_call:
                        # Another function received, old function complete, can break here.
                        break
                    func_call = {
                        "id": tool_calls.id,
                        "func_name": tool_calls.function.name or "",
                        "arguments": {},
                    }
                else:
                    # append argument
                    func_arguments += tool_calls.function.arguments or ""

            # Parse transcripts
            if chunk.choices[0].delta.content:
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=chunk.choices[0].delta.content,
                    content_complete=False,
                    end_call=False,
                )
                yield response

        # Step 4: Call the functions
        if func_call:
            function_name = func_call["func_name"]
            if function_name == "end_call":
                func_call["arguments"] = json.loads(func_arguments)
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=func_call["arguments"]["message"],
                    content_complete=True,
                    end_call=True,
                )
                yield response

            elif function_name == "check_availability":
                available_times = check_availability(function_args['date'])
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=f"Available times on {function_args['date']}: {', '.join(available_times)}",
                    content_complete=True,
                    end_call=False,
                )
                yield response
            elif function_name == "book_appointment":
                success = book_appointment(
                    function_args['date'],
                    function_args['time'],
                    function_args['customer_name'],
                    function_args['customer_email'],
                    function_args['customer_phone']
                )
                response = ResponseResponse(
                    response_id=request.response_id,
                    content="Appointment booked successfully!" if success else "Sorry, that time slot is not available.",
                    content_complete=True,
                    end_call=False,
                )
                yield response
        else:
            # No functions, complete response
            response = ResponseResponse(
                response_id=request.response_id,
                content="",
                content_complete=True,
                end_call=False,
            )
            yield response
