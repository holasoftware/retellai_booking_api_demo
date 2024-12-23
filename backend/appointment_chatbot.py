import logging
import datetime


from llm_fsm import ConversationalLLMStateMachine

logger = logging.getLogger(__name__)


BEGIN_SENTENCE = "Hey there, I'm your personal hair salon assistant, how can I help you?"
GENERIC_GOAL = """You are assisting customers with inquiries about our hair salon "Filpino haircuts". Your answer their questions about services and pricing, and schedule appointments. This is the information regarding the hair salon:
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
- appointment: the user wants to make an appointment
- appointment_date_availability: the user wants an appointment for a specific date
- appointment_providing_information: the user is providing information about the contact information
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
                    "enum": ["appointment", "appointment_date_availability", "appointment_confirmation", "appointment_providing_information", "information_inquiry", "service_complain", "thanks"]
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
        "description": "The user is asking to schedule an appointment for the service. Optionally the user is providing a date",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "The date the user is asking for the appointment, in YYYY-MM-DD format"
                }
            }
        }
    }
}

appointment_confirmation_tool = {
    "type": "function",
    "function": {
        "name": "appointment_confirmation",
        "description": "The user confirms that the date and time for the appointment and the  and contact information are correct",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Message to answer to the user"
            }
        },
        "required": ["answer"]
    }
}

schedule_appointment_tool = {
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
        "required": ["date", "time", "customer_name", "customer_phone"]
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

state_machine = ConversationalLLMStateMachine(default_llm_model="gpt-4o-mini", common_tools=[end_call_tool, detect_user_intent_tool])

@state_machine.define_state(goal=GENERIC_GOAL, tools=[detect_user_intent_tool, schedule_appointment_tool], responses_per_user_intent=[
    {
        "user_intent": "to make an appointment",
        "answer": "ask first for a date for the appointment that best suits the customer "
    }
])
def general_inquiries(data):
    tools = data["tools"]

    if "schedule_appointment" in tools:
        if "date" in tools["schedule_appointment"]:
            return "appointment_show_date_time_slots"
        else:
            return "appointment_select_date"
    else:
        return "general_inquiries"


def appointment_select_date_system_message(data):
    today = datetime.datetime.now().date()

    system_message="Ask the user to select a date for the appointment. Today is {today}".format(today=today)
    return system_message

@state_machine.define_state(goal=GENERIC_GOAL, system_message=appointment_select_date_system_message, tools=[check_appointment_date_availability_tool])
def appointment_select_date(data):
    pass


def appointment_show_date_time_slots_system_message(data):
    appointment_date = data["appointment_date"]
    time_slots = check_availability(date)

    if len(time_slots) == 0:
        extra_system_message = "Tell to the user that there is no available time slot for this date  %s and ask for a different date" % appointment_date
    else:
        extra_system_message = "Ask to the user to select one of the available time slots for the date %s:\n%s" % (appointment_date, "\n".join(["%d. %s" % (i, time_slot) for (i, time_slot) in enumerate(time_slots)])

    return extra_system_message


@state_machine.define_state(goal=GENERIC_GOAL, system_message=appointment_show_date_time_slots_system_message)
def appointment_show_date_time_slots(data):
    pass

@state_machine.define_state(goal=GENERIC_GOAL, system_message="Ask for contact information necessary for the appointment: name, phone number and optionally email", tools=[schedule_appointment_tool])
def appointment_collect_contact_information(data):
    pass

def appointment_confirm_system_message(data):
    system_message = "Confirm the appointment: {appointment_information}"
    return system_message

@state_machine.define_state(goal=GENERIC_GOAL, system_message=appointment_confirm_system_message, tools=[appointment_confirmation_tool])
def appointment_confirm(data):
    pass
