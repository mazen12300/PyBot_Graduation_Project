"""
generate_dataset.py
--------------------
Builds the labeled intent dataset for the "College Subjects & FAQ Chatbot".

Each row = one example student message + the intent (category) it belongs to.
Two families of intents:
  1) Administrative / FAQ intents  -> answered directly from a fixed response
     (greeting, goodbye, thanks, exam_schedule, assignment_deadline,
      office_hours, course_registration, grading_system, library_hours,
      technical_support)
  2) Academic / tutoring intents   -> forwarded to the LLM tutor for a full
     explanation (python_basics, data_structures, databases,
     machine_learning, statistics, general_help)

The dataset is intentionally imbalanced across intents (8 to 22 examples)
so the preprocessing pipeline has a real imbalance problem to detect and fix.
"""

import pandas as pd
import os

DATA = {
    "greeting": [
        "Hi", "Hello", "Hey there", "Good morning", "Good evening",
        "Hi there, how are you?", "Hello, is anyone here?", "Hey, can you help me?",
        "Good afternoon", "Hi, I need some help", "Hello there", "Hey",
        "Greetings", "Hi assistant", "Good day",
    ],
    "goodbye": [
        "Bye", "Goodbye", "See you later", "I have to go now", "Talk to you later",
        "See you", "Bye bye", "I'm leaving now", "Catch you later",
        "See you next time", "Goodbye for now", "Thanks, bye",
    ],
    "thanks": [
        "Thank you", "Thanks a lot", "Thank you so much", "I appreciate it",
        "Thanks for the help", "That was helpful, thanks", "Thank you very much",
        "Thanks!", "Much appreciated", "Thanks for your time",
        "I really appreciate your help", "Great, thank you",
    ],
    "python_basics": [
        "What is a variable in Python?", "How do I install Python?",
        "What is the difference between a list and a tuple?",
        "How do I write a for loop in Python?", "What is a function in Python?",
        "Explain Python data types", "How do I read a file in Python?",
        "What is indentation in Python?", "How do I install a Python package?",
        "What is a dictionary in Python?", "How do I handle exceptions in Python?",
        "What is object-oriented programming in Python?",
        "How do I define a class in Python?", "What is a lambda function?",
        "How do virtual environments work in Python?", "What is pip used for?",
        "How do I import a module in Python?",
        "What is the difference between == and is in Python?",
        "How do I debug Python code?", "What is a Python decorator?",
        "How do list comprehensions work?", "What is the GIL in Python?",
    ],
    "data_structures": [
        "What is a stack?", "What is a queue?", "Explain how a linked list works",
        "What is the difference between an array and a linked list?",
        "What is a binary search tree?", "How does a hash table work?",
        "What is the time complexity of quicksort?", "What is a graph data structure?",
        "Explain breadth first search", "Explain depth first search",
        "What is a heap?", "How do I implement a stack using an array?",
        "What is Big O notation?", "What is recursion?",
        "Explain how binary search works", "What is the difference between BFS and DFS?",
        "What is a balanced binary tree?", "How does merge sort work?",
        "What is dynamic programming?", "What is a doubly linked list?",
    ],
    "databases": [
        "What is SQL?", "What is the difference between SQL and NoSQL?",
        "How do I write a JOIN query?", "What is a primary key?",
        "What is a foreign key?", "What is database normalization?",
        "What is an index in a database?", "How do I create a table in SQL?",
        "What is a stored procedure?", "What is ACID in databases?",
        "What is the difference between INNER JOIN and OUTER JOIN?",
        "How do transactions work in a database?", "What is a database schema?",
        "What is denormalization?", "How do I optimize a slow SQL query?",
        "What is MongoDB?",
        "What is the difference between relational and non-relational databases?",
        "What is a composite key?",
    ],
    "machine_learning": [
        "What is machine learning?",
        "What is the difference between supervised and unsupervised learning?",
        "What is overfitting?", "What is underfitting?", "What is a confusion matrix?",
        "What is cross-validation?", "What is gradient descent?",
        "What is a neural network?",
        "What is the difference between classification and regression?",
        "What is feature engineering?", "What is regularization in machine learning?",
        "What is a random forest?", "What is the bias-variance tradeoff?",
        "What is a support vector machine?", "How does k-means clustering work?",
        "What is precision and recall?", "What is deep learning?",
        "What is transfer learning?", "What is an activation function?",
        "What is a loss function?",
    ],
    "statistics": [
        "What is the mean and median?", "What is standard deviation?",
        "What is a p-value?", "What is a normal distribution?",
        "What is correlation?", "What is hypothesis testing?",
        "What is the central limit theorem?", "What is variance?",
        "What is a confidence interval?",
        "What is the difference between correlation and causation?",
        "What is a t-test?", "What is skewness?", "What is a chi-square test?",
        "What is sampling in statistics?", "What is a probability distribution?",
        "What is Bayes theorem?",
    ],
    "exam_schedule": [
        "When is the next exam?", "What is the exam schedule for this semester?",
        "When is the Python exam?", "Can you tell me the final exam dates?",
        "When are midterm exams?", "Is there an exam next week?",
        "What time does the exam start?", "Where can I check the exam timetable?",
        "When is the machine learning exam?", "Are exams online or in person?",
        "How long is the final exam?", "When will exam results be announced?",
        "Is there a makeup exam?", "How many exams are left this term?",
    ],
    "assignment_deadline": [
        "When is the assignment due?", "What is the deadline for the project?",
        "Can I get an extension on my assignment?",
        "When do I need to submit my homework?",
        "Is there a late submission policy?",
        "What is the deadline for the machine learning project?",
        "How do I submit my assignment?", "When is the final project due?",
        "Can I submit my assignment late?", "What happens if I miss the deadline?",
        "How many days do I have to finish this assignment?",
        "What is the submission format for the project?",
        "Where do I upload my assignment?", "Is the deadline the same for everyone?",
    ],
    "office_hours": [
        "What are the professor's office hours?", "When can I meet the instructor?",
        "How do I book office hours?", "Are office hours available online?",
        "What days are office hours held?", "Can I schedule a one-on-one meeting?",
        "Where are office hours held?", "How long are office hours?",
        "Do I need an appointment for office hours?", "Are office hours mandatory?",
    ],
    "course_registration": [
        "How do I register for a course?",
        "What is the deadline to register for courses?",
        "Can I change my course schedule?", "How do I drop a course?",
        "What are the prerequisites for this course?",
        "How many courses can I register for?", "Is there a registration fee?",
        "Can I register for a course after the deadline?",
        "How do I check my registered courses?", "What is the add/drop period?",
        "Can I switch to a different track?", "How do I register for an elective?",
    ],
    "grading_system": [
        "How is the final grade calculated?", "What is the passing grade?",
        "How much is the project worth in the final grade?",
        "Is attendance part of the grade?", "What is the grading scale?",
        "How are assignments graded?", "Can I see my grades online?",
        "Is there a curve on the exam grades?", "How do I appeal a grade?",
        "What percentage is the final exam worth?",
    ],
    "library_hours": [
        "What time does the library open?", "Is the library open on weekends?",
        "What are the library hours?", "Can I borrow books from the library?",
        "How long can I keep a borrowed book?", "Is there a digital library available?",
        "Does the library have study rooms?", "What time does the library close?",
    ],
    "technical_support": [
        "I can't access the online platform", "The website is not loading",
        "I forgot my password", "How do I reset my account password?",
        "The video lecture is not playing",
        "I'm getting an error when I submit my assignment",
        "My account is locked", "I can't log into the portal",
    ],
    "general_help": [
        "Can you help me with something?", "I have a question",
        "I need some assistance", "Can you explain something to me?",
        "I don't understand this topic", "Can you help me study?",
        "I'm confused about something", "What can you help me with?",
        "Can you give me some advice?", "I need help understanding a concept",
    ],
}

# Canned responses for the administrative / FAQ intents (fast path, no LLM call)
FAQ_RESPONSES = {
    "greeting": "Hi! I'm PyBot 🤖. Ask me about your courses, exams, deadlines, or any Python/AI topic!",
    "goodbye": "Goodbye! Good luck with your studies 👋",
    "thanks": "You're very welcome! Happy to help anytime 😊",
    "exam_schedule": "Exam schedules are published on the college portal under 'Exams > Timetable'. Please check there for the exact date, time, and room for each subject.",
    "assignment_deadline": "Assignment deadlines are listed on the course page on the LMS. Late submissions may be penalized unless you have an approved extension — check with your instructor.",
    "office_hours": "Office hours are usually held twice a week and can be booked through the instructor's booking link on the LMS. Check your course page for the exact times.",
    "course_registration": "You can register for courses through the student portal during the official registration window. Make sure you meet the listed prerequisites before registering.",
    "grading_system": "Final grades are usually a weighted combination of assignments, exams, and participation. The exact breakdown is listed in each course's syllabus.",
    "library_hours": "The library is typically open from 9:00 AM to 8:00 PM on weekdays, with reduced hours on weekends. Check the library page for holiday schedules.",
    "technical_support": "Sorry you're having trouble! Please try clearing your browser cache or resetting your password from the login page. If the issue continues, open a support ticket with IT.",
}

rows = []
for intent, examples in DATA.items():
    for text in examples:
        rows.append({"text": text, "intent": intent})

df = pd.DataFrame(rows)

out_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(out_dir, "college_chatbot_dataset.csv")
df.to_csv(csv_path, index=False)

resp_path = os.path.join(out_dir, "faq_responses.csv")
pd.DataFrame(
    [{"intent": k, "response": v} for k, v in FAQ_RESPONSES.items()]
).to_csv(resp_path, index=False)

print(f"Dataset saved to: {csv_path}  ({len(df)} rows, {df['intent'].nunique()} intents)")
print(f"FAQ responses saved to: {resp_path}")
print(df["intent"].value_counts())
