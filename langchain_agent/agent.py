import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# =========================================================

# Load variables stored inside the .env file.
load_dotenv()


# Read the Gemini API key from the environment.
gemini_api_key = os.getenv("GEMINI_API_KEY")


# Stop the program if the API key is unavailable.
if not gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Please add GEMINI_API_KEY inside your .env file."
    )


# =========================================================
# TOOL 1: GET COURSE DETAILS
# =========================================================

@tool
def get_course_detail(detail_name: str) -> str:
    """
    Get exact information about the AI course.

    Use this tool when the user asks about the course fee,
    duration, mentor, mode or placement assistance.

    Args:
        detail_name: The course detail requested by the user.
                     Valid values are fee, duration, mentor,
                     mode and placement.

    Returns:
        The requested course information.
    """

    course_data = {
        "fee": "45000",
        "duration": "12 weeks",
        "mentor": "Hassan",
        "mode": "Online",
        "placement": "Placement assistance is available"
    }

    # Convert the model input into lowercase and remove spaces.
    cleaned_name = detail_name.lower().strip()

    # Return the requested value when it exists.
    if cleaned_name in course_data:
        return course_data[cleaned_name]

    return (
        "Detail not found. Available details are "
        "fee, duration, mentor, mode and placement."
    )


# =========================================================
# TOOL 2: ADD PERCENTAGE
# =========================================================

@tool
def add_percentage(
    amount: float,
    percentage: float
) -> float:
    """
    Add a percentage to an amount.

    Use this tool for GST, tax, markup or any percentage
    increase.

    Args:
        amount: The original amount.
        percentage: The percentage that must be added.

    Returns:
        The amount after adding the percentage.
    """

    percentage_amount = amount * percentage / 100

    final_amount = amount + percentage_amount

    return round(final_amount, 2)


# =========================================================
# TOOL 3: CALCULATE DISCOUNT
# =========================================================

@tool
def calculate_discount(
    amount: float,
    discount_percentage: float
) -> float:
    """
    Subtract a discount percentage from an amount.

    Use this tool when the user asks for a discounted price.

    Args:
        amount: The original amount.
        discount_percentage: The percentage discount.

    Returns:
        The amount remaining after the discount.
    """

    discount_amount = amount * discount_percentage / 100

    final_amount = amount - discount_amount

    return round(final_amount, 2)


# =========================================================
# TOOL 4: CALCULATE INSTALLMENT
# =========================================================

@tool
def calculate_installment(
    total_amount: float,
    number_of_installments: int
) -> float:
    """
    Divide a total amount into equal installments.

    Use this tool when the user asks for monthly, weekly
    or equal payment installments.

    Args:
        total_amount: The complete amount to be divided.
        number_of_installments: Number of equal payments.

    Returns:
        The amount of one installment.
    """

    if number_of_installments <= 0:
        return 0.0

    installment_amount = total_amount / number_of_installments

    return round(installment_amount, 2)


# =========================================================
# TOOL 5: CALCULATE TOTAL FEE FOR STUDENTS
# =========================================================

@tool
def calculate_total_fee(
    fee_per_student: float,
    number_of_students: int
) -> float:
    """
    Calculate the total fee for multiple students.

    Use this tool to multiply the fee per student by the
    number of students.

    Args:
        fee_per_student: Course fee for one student.
        number_of_students: Total number of students.

    Returns:
        Total fee for all students.
    """

    if number_of_students < 0:
        return 0.0

    total_fee = fee_per_student * number_of_students

    return round(total_fee, 2)


# =========================================================
# TOOL 6: CALCULATE AVERAGE MARKS
# =========================================================

@tool
def calculate_average(
    mark1: float,
    mark2: float,
    mark3: float
) -> float:
    """
    Calculate the average of three subject marks.

    Args:
        mark1: Marks obtained in the first subject.
        mark2: Marks obtained in the second subject.
        mark3: Marks obtained in the third subject.

    Returns:
        Average of the three marks.
    """

    total_marks = mark1 + mark2 + mark3

    average_marks = total_marks / 3

    return round(average_marks, 2)


# =========================================================
# TOOL 7: GET STUDENT GRADE
# =========================================================

@tool
def get_grade(score: float) -> str:
    """
    Return a student's grade based on a score from 0 to 100.

    Use this tool after receiving or calculating a student's
    marks or average score.

    Args:
        score: Student score between 0 and 100.

    Returns:
        Student grade.
    """

    if score < 0 or score > 100:
        return "Invalid score. Score must be between 0 and 100."

    if score >= 90:
        return "A+"

    if score >= 80:
        return "A"

    if score >= 70:
        return "B"

    if score >= 60:
        return "C"

    if score >= 50:
        return "D"

    return "Fail"


# =========================================================
# TOOL 8: CHECK COURSE ELIGIBILITY
# =========================================================

@tool
def check_course_eligibility(
    age: int,
    has_basic_python: bool
) -> str:
    """
    Check whether a student is eligible for the AI course.

    A student must be at least 18 years old and must know
    basic Python.

    Args:
        age: Age of the student.
        has_basic_python: Whether the student knows basic
                          Python.

    Returns:
        Eligibility result.
    """

    if age < 18:
        return "Not eligible because the minimum age is 18."

    if not has_basic_python:
        return (
            "Not eligible yet. Complete basic Python before "
            "joining the course."
        )

    return "The student is eligible for the AI course."


# =========================================================
# TOOL 9: CONVERT WEEKS TO DAYS
# =========================================================

@tool
def convert_weeks_to_days(weeks: float) -> float:
    """
    Convert weeks into days.

    Use this tool when a duration is provided in weeks but
    the user wants the duration in days.

    Args:
        weeks: Number of weeks.

    Returns:
        Equivalent number of days.
    """

    days = weeks * 7

    return round(days, 2)


# =========================================================
# TOOL 10: COUNT WORDS
# =========================================================

@tool
def count_words(text: str) -> int:
    """
    Count the total number of words in a text.

    Args:
        text: Sentence or paragraph whose words must be
              counted.

    Returns:
        Total number of words.
    """

    words = text.split()

    return len(words)


# =========================================================
# STEP 2: CREATE THE GEMINI MODEL
# =========================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    api_key=gemini_api_key
)


# =========================================================
# STEP 3: STORE ALL TOOLS IN A LIST
# =========================================================

# Because these functions use @tool, every item in this list
# is now a LangChain Tool object rather than a normal Python
# function.

tools = [
    get_course_detail,
    add_percentage,
    calculate_discount,
    calculate_installment,
    calculate_total_fee,
    calculate_average,
    get_grade,
    check_course_eligibility,
    convert_weeks_to_days,
    count_words
]


# =========================================================
# STEP 4: CREATE THE AGENT
# =========================================================

agent = create_agent(
    model=model,

    tools=tools,

    system_prompt="""
You are a beginner-friendly AI course and student assistant.

You have access to the following tools:

1. get_course_detail
   Use it for the course fee, duration, mentor, mode or
   placement information.

2. add_percentage
   Use it for GST, tax, markup or percentage increases.

3. calculate_discount
   Use it when a discount must be subtracted.

4. calculate_installment
   Use it when an amount must be divided into equal
   installments.

5. calculate_total_fee
   Use it when calculating the total fee for multiple
   students.

6. calculate_average
   Use it to calculate the average of three marks.

7. get_grade
   Use it to determine a grade from a score or average.

8. check_course_eligibility
   Use it to check whether a student can join the course.

9. convert_weeks_to_days
   Use it to convert a duration from weeks to days.

10. count_words
    Use it to count words in a sentence or paragraph.

Important rules:

- Use the available tools whenever an exact calculation or
  stored course information is required.

- You may call multiple tools for one user question.

- Use the result of one tool as input to another tool when
  necessary.

- Do not guess course information.

- Do not manually perform a calculation when a suitable
  calculation tool is available.

- Give the final answer clearly and briefly.
"""
)


# =========================================================
# STEP 5: TAKE INPUT FROM THE USER
# =========================================================

question = input("\nEnter your question: ")


# =========================================================
# STEP 6: RUN THE AGENT
# =========================================================

# agent.invoke() runs the complete agent loop:
#
# User message
#      ↓
# Model decision
#      ↓
# Tool call, when required
#      ↓
# Tool result
#      ↓
# Model receives tool result
#      ↓
# Final answer
#
# The returned result is the final agent state.
# Its "messages" field contains all messages generated during
# the execution.

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    },

    config={
        # Prevent the model and tools from running in an
        # unlimited loop.
        "recursion_limit": 20
    }
)


# =========================================================
# STEP 7: ACCESS THE FINAL MESSAGE
# =========================================================

# result["messages"] contains:
#
# HumanMessage -> original question
# AIMessage    -> model decision or final response
# ToolMessage  -> result returned by a tool
#
# The last message is normally the final AI response.

final_message = result["messages"][-1]


# =========================================================
# STEP 8: PRINT ONLY THE FINAL ANSWER
# =========================================================

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(final_message.content) 