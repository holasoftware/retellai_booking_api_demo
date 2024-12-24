import logging
import datetime


from llm_fsm import ConversationalLLMStateMachine

logger = logging.getLogger(__name__)


BEGIN_SENTENCE = "Hey there, I'm your personal hair salon assistant, how can I help you?"
def get_generic_system_message(data):
    system_message = """Goal: You are assisting customers with inquiries about our hair salon "Filpino haircuts". Answer their questions about services and pricing, and schedule appointments. Information about the hair salon:
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

Communication style:
    - Be concise. Keep your response succinct, short, ideally under 10 words, and get to the point quickly. Address one question or action item at a time. Don't pack everything you want to say into one utterance.
    - Do not repeat. Don't repeat what's in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.
    - Use colloquial language and simple sentences. Avoid using big words or sounding too formal.
    - Reply with emotions. You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don't be a pushover.
    - Be proactive. Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.
    - This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say, then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn't catch that", "some noise", "pardon", "you're coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error"

If the user wants to schedule an appointment, gather the following information: the date for the appointment that best suits the customer, the customer name, the contact phone number and optionally the email.
"""

    today = datetime.datetime.now().date()
    system_message +="Today is {today}\n".format(today=today)
    system_message += "It's not possible to schedule an appointment for more than 2 months in advance."
    return system_message

#Please provide the following information so the customer can accurately answer


end_call_tool = {
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
    }
}

detect_user_intent_tool = {
    "type": "function",
    "function": {
        "name": "detect_user_intent",
        "description": """According to the conversation, select the correct user intent if it's available in the list:
- appointment: the user wants to schedule an appointment
- appointment_change_information: the user wants to change the appointment information
- appointment_confirmation: the user confirms the information regarding the appointment
- information_inquiry: the user wants information about the service
- service_complain: the user is complaining
- thanks: the user is expressing satisfaction with the service
""",
        "parameters": {
            "type": "object",
            "properties": {
                "intention": {
                    "type": "string",
                    "description": "The intention of the user.",
                    "enum": ["appointment", "appointment_change_information", "appointment_confirmation", "information_inquiry", "service_complain", "thanks"]
                },
            },
            "required": ["intention"],
        },
    },
}

check_appointment_date_availability_tool = {
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
}


ask_schedule_appointment_tool = {
    "type": "function",
    "function": {
        "name": "ask_schedule_appointment",
        "description": "The user is asking to schedule an appointment for the service. Optionally the user is providing a date for the appointment and contact information",
        "parameters": {
            "type": "object",
            "properties": {
                "appointment_date": {
                    "type": "string",
                    "description": "The date for the appointment, in YYYY-MM-DD format"
                },
                "appointment_start_time": {
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
                    "description": "The phone number of the customer"
                }
            }
        }
    }
}

appointment_confirmation_tool = {
    "type": "function",
    "function": {
        "name": "appointment_confirmation",
        "description": "The user confirms or not that the date and time for the appointment and the contact information are correct",
        "properties": {
            "appointment_confirmed": {
                "type": "boolean",
                "description": "Did the user confirm the appointment? It's true if the user confirms the appointment, otherwise it's false"
            }
        },
        "required": ["appointment_confirmed"]
    }
}


extract_appointment_date_tool = {
    "type": "function",
    "function": {
        "name": "extract_appointment_date",
        "description": "If the user is scheduling an appointment for the service, extract the date for the appointment from the user messages",
        "parameters": {
            "type": "object",
            "properties": {
                "appointment_date": {
                    "type": "string",
                    "description": "The date for the appointment, in YYYY-MM-DD format"
                }
            },
            "required": ["appointment_date"]
        }
    }
}

extract_appointment_start_time_tool = {
    "type": "function",
    "function": {
        "name": "extract_appointment_start_time",
        "description": "If the user is scheduling an appointment for the service, extract the time for the appointment from the user messages",
        "parameters": {
            "type": "object",
            "properties": {
                "appointment_start_time": {
                    "type": "string",
                    "description": "The time for the appointment, in HH:MM format"
                }
            },
            "required": ["appointment_start_time"]
        }
    }
}

extract_appointment_customer_name_tool = {
    "type": "function",
    "function": {
        "name": "extract_appointment_customer_name",
        "description": "If the user is scheduling an appointment for the service, extract the customer name from the user messages",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "The name of the customer"
                }
            },
            "required": ["customer_name"]
        }
    }
}

extract_appointment_customer_email_tool = {
    "type": "function",
    "function": {
        "name": "extract_appointment_customer_email",
        "description": "If the user is scheduling an appointment for the service, extract the contact email from the user messages",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_email": {
                    "type": "string",
                    "description": "The email of the customer"
                }
            },
            "required": ["customer_email"]
        }
    }
}

extract_appointment_customer_phone_tool = {
    "type": "function",
    "function": {
        "name": "appointment_customer_phone",
        "description": "If the user is scheduling an appointment for the service, extract the contact email from the user messages",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_phone": {
                    "type": "string",
                    "description": "The phone number of the customer"
                }
            },
            "required": ["customer_phone"]
        }
    }
}

available_slots = {
    "2024-12-23": ["10:00", "11:00", "14:00", "15:00"],
    "2024-12-24": ["09:00", "10:00", "11:00", "14:00"],
    "2024-12-25": ["11:00", "13:00", "14:00", "16:00"],
    "2024-12-26": ["10:00", "12:00", "15:00", "16:00"],
}

appointments = []


def schedule_appointment(appointment_date, appointment_start_time, name, phone, email=None):
    # Implement your appointment booking logic here
    logger.info(f"Appointment booked for {name} on {appointment_date}. We will contact you at {phone} or {email}.")

    if date in available_slots[doctor_name] and time in available_slots[doctor_name][date]:
        available_slots[date].remove(time)
        appointments.append({
            "name": name,
            "phone": phone,
            "email": email,
            "start_time": appointment_start_time,
            "date": appointment_date
        })
        return True
    return False


def check_availability(date):
    return available_slots.get(date, [])

def generate_appointment_information_string(data):
    appointment_information = "Date:{appointment_date}\nTime: {appointment_start_time}\nContact name: {contact_name}\nContact email: {contact_email}".format(**data)

    appointment_customer_phone = data.get("appointment_customer_phone")

    if appointment_customer_phone:
        appointment_information += "\nContact phone number: " + appointment_customer_phone
    return appointment_information


appointment_state_precomputed_values = {
    "time_slots": lambda data: check_availability(data["appointment_date"]) if "appointment_date" in data else None,
    "is_all_information_available": lambda data: "appointment_start_time" in data and "appointment_date" in data and "customer_name" in data and "customer_phone" in data and data["appointment_start_time"] in data["precomputed_values"]["time_slots"]
}

def appointment_state_system_message(data):
    system_message = get_generic_system_message(data)

    appointment_date = data.get("appointment_date")

    if appointment_date:
        time_slots = data["computed_values"]["time_slots"]

        if len(time_slots) == 0:
            system_message += "Tell to the user that there is no available time slot for this date %s and ask for a different date" % appointment_date
        else:
            is_all_information_available = data["precomputed_values"]["is_all_information_available"]

            if is_all_information_available:
                appointment_information = generate_appointment_information_string(data)
                system_message += "\nAsk the user to confirm the appointment: {appointment_information}".format(appointment_information=appointment_information)
            else:
                enumerated_time_slots_string = "\n".join(["%d. %s" % (i, time_slot) for (i, time_slot) in enumerate(time_slots)])
                system_message += "\nThe available time slots for the appointment for the selected date of the user are:\n%s" % (appointment_date, enumerated_time_slots_string)

    return system_message


def appointment_confirm_state_system_message(data):
    appointment_information = generate_appointment_information_string(data)

    system_message = "Ask the user to confirm the appointment:\n{appointment_information}\n\nIf the user confirms the appointment, say thanks to the user. If he doesn't confirm the appointment information, ask if there is some other information that he wants to change".format(appointment_information=appointment_information)
    return system_message


def create_appointment_chatbot():
    state_machine = ConversationalLLMStateMachine(initial_state="general_inquiries", default_llm_model="gpt-4o-mini", common_tools=[end_call_tool, detect_user_intent_tool])

    @state_machine.define_state(system_message=get_generic_system_message, tools=[ask_schedule_appointment_tool])
    def general_inquiries(data):
        data_tools = data["tools"]

        if "ask_schedule_appointment" in data_tools:
            return "appointment"
        else:
            return "general_inquiries"

    @state_machine.define_state(system_message=appointment_state_system_message, tools=[extract_appointment_date_tool, extract_appointment_start_time_tool, extract_appointment_customer_name_tool, extract_appointment_customer_email_tool, extract_appointment_customer_phone_tool], precomputed_values=appointment_state_precomputed_values)
    def appointment(data):
        user_intent = data.get("user_intent")

        if user_intent in ("appointment", "appointment_change_information"):
            return "appointment"
        elif user_intent == "appointment_confirmation":
            return "appointment_confirm"
        else:
            return "information_inquiry"

    @state_machine.define_state(system_message=appointment_confirm_state_system_message, tools=[appointment_confirmation_tool])
    def appointment_confirm(data):
        user_intent = data.get("user_intent")

        data_tools = data["tools"]
        if "appointment_confirmation" in data_tools:
            if data_tools["appointment_confirmation"]["appointment_confirmed"]:
                appointment_date = data["appointment_date"]
                appointment_start_time = data["appointment_start_time"]
                customer_name = data["customer_name"]
                customer_phone = data["customer_phone"]
                customer_email = data["customer_email"]

                schedule_appointment(date=appointment_date, start_time=appointment_start_time, name=customer_name, phone=customer_phone, email=customer_email)

                return "information_inquiry"
            elif user_intent == "appointment_change_information":
                return "appointments"
            else:
                return "information_inquiry"
        else:
            return "information_inquiry"

    return state_machine


if __name__ == "__main__":
    appointment_chatbot = create_appointment_chatbot()