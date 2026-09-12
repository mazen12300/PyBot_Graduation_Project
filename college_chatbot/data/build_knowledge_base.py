"""
build_knowledge_base.py
------------------------
Creates the RAG knowledge base: short "lecture note" chunks for each
academic subject PyBot tutors on. This is the content the retriever
searches over before the LLM generates an answer, so PyBot's academic
answers are grounded in real course material instead of just the LLM's
general knowledge.

Replace/extend these chunks with your actual course PDFs/slides content
whenever you have them — the structure (subject, title, text) is all the
rest of the pipeline needs.
"""

import json
import os

KB = [
    # ---------------- Python Basics ----------------
    {
        "subject": "python_basics", "title": "Variables and Data Types",
        "text": (
            "In Python, a variable is a name that refers to a value stored in memory. "
            "You don't need to declare a type explicitly — Python infers it at runtime "
            "(dynamic typing). The core built-in types are: int (whole numbers), "
            "float (decimal numbers), str (text), bool (True/False), list (ordered, "
            "mutable collection), tuple (ordered, immutable collection), dict "
            "(key-value pairs), and set (unordered, unique values). Example: "
            "age = 20; name = 'Ali'; is_student = True."
        ),
    },
    {
        "subject": "python_basics", "title": "Control Flow: if/for/while",
        "text": (
            "Python uses indentation (not braces) to define code blocks. An if "
            "statement runs code conditionally: if x > 0: print('positive'). A for "
            "loop iterates over a sequence: for item in my_list: print(item). A while "
            "loop repeats while a condition is true: while count < 5: count += 1. "
            "Use break to exit a loop early and continue to skip to the next iteration."
        ),
    },
    {
        "subject": "python_basics", "title": "Functions",
        "text": (
            "A function is a reusable block of code defined with the def keyword: "
            "def add(a, b): return a + b. Functions can have default parameter values "
            "(def greet(name='friend')), accept variable arguments (*args, **kwargs), "
            "and return multiple values as a tuple. A lambda is a small anonymous "
            "function: square = lambda x: x * x."
        ),
    },
    {
        "subject": "python_basics", "title": "Lists, Tuples, and Dictionaries",
        "text": (
            "A list ([1, 2, 3]) is ordered and mutable — you can add, remove, or "
            "change items. A tuple ((1, 2, 3)) is ordered but immutable — once "
            "created it cannot change, which makes it slightly faster and hashable. "
            "A dictionary ({'key': 'value'}) stores key-value pairs and gives O(1) "
            "average lookup time by key. List comprehensions offer a compact way to "
            "build lists: squares = [x**2 for x in range(10)]."
        ),
    },
    {
        "subject": "python_basics", "title": "Object-Oriented Programming in Python",
        "text": (
            "A class is a blueprint for creating objects. It's defined with class "
            "ClassName: and typically has an __init__ method (the constructor) that "
            "sets up instance attributes: def __init__(self, name): self.name = name. "
            "Methods are functions defined inside a class that operate on an "
            "instance. Python supports inheritance (class Dog(Animal):), letting a "
            "subclass reuse and extend a parent class's behavior."
        ),
    },
    {
        "subject": "python_basics", "title": "Exception Handling",
        "text": (
            "Python handles runtime errors with try/except blocks: try: risky_code() "
            "except ValueError as e: print(e). You can catch multiple exception "
            "types, use a generic except Exception as a fallback, use finally to run "
            "cleanup code regardless of success or failure, and raise your own "
            "exceptions with raise ValueError('message') for validation."
        ),
    },
    {
        "subject": "python_basics", "title": "Modules, Packages, and pip",
        "text": (
            "A module is a single .py file; a package is a folder of modules with an "
            "__init__.py file. You import code with import module_name or "
            "from module_name import function_name. pip is Python's package "
            "manager — pip install package_name downloads and installs third-party "
            "libraries from PyPI. Virtual environments (python -m venv venv) isolate "
            "a project's dependencies from the system Python installation."
        ),
    },

    # ---------------- Data Structures ----------------
    {
        "subject": "data_structures", "title": "Arrays vs. Linked Lists",
        "text": (
            "An array stores elements in contiguous memory, giving O(1) random "
            "access by index but O(n) insertion/deletion in the middle (elements "
            "must shift). A linked list stores elements as nodes, each pointing to "
            "the next, giving O(1) insertion/deletion at a known position but O(n) "
            "access by index since you must traverse from the head. Arrays are more "
            "cache-friendly; linked lists are more flexible in size."
        ),
    },
    {
        "subject": "data_structures", "title": "Stacks and Queues",
        "text": (
            "A stack is a Last-In-First-Out (LIFO) structure — the last element "
            "pushed is the first one popped. Common uses: undo functionality, "
            "expression evaluation, call stacks in recursion. A queue is a "
            "First-In-First-Out (FIFO) structure — the first element enqueued is "
            "the first dequeued. Common uses: task scheduling, breadth-first search, "
            "print job queues. Both support O(1) insertion and removal at their "
            "respective ends when implemented well (e.g., with a linked list or a "
            "circular buffer)."
        ),
    },
    {
        "subject": "data_structures", "title": "Binary Search Trees",
        "text": (
            "A binary search tree (BST) is a tree where each node's left subtree "
            "contains only smaller values and its right subtree only larger values. "
            "This property enables O(log n) average-case search, insertion, and "
            "deletion by repeatedly halving the search space — similar to binary "
            "search on a sorted array. However, an unbalanced BST (e.g., built from "
            "sorted input) can degrade to O(n), which is why self-balancing trees "
            "like AVL or Red-Black trees exist."
        ),
    },
    {
        "subject": "data_structures", "title": "Hash Tables",
        "text": (
            "A hash table maps keys to values using a hash function that converts a "
            "key into an array index, giving average O(1) insertion, lookup, and "
            "deletion. Collisions (two keys hashing to the same index) are handled "
            "via chaining (a linked list per bucket) or open addressing (probing for "
            "the next free slot). Python's dict is implemented as a hash table."
        ),
    },
    {
        "subject": "data_structures", "title": "Big O Notation and Recursion",
        "text": (
            "Big O notation describes how an algorithm's running time or memory use "
            "grows as input size increases, ignoring constant factors. Common "
            "classes from fastest to slowest: O(1), O(log n), O(n), O(n log n), "
            "O(n^2), O(2^n). Recursion is when a function calls itself to solve a "
            "smaller instance of the same problem, always needing a base case to "
            "stop; every recursive solution can also be rewritten iteratively using "
            "an explicit stack."
        ),
    },
    {
        "subject": "data_structures", "title": "Sorting: Merge Sort and Quicksort",
        "text": (
            "Merge sort splits the array in half recursively, sorts each half, then "
            "merges them — guaranteed O(n log n) time but needs O(n) extra space. "
            "Quicksort picks a pivot, partitions elements smaller/larger than the "
            "pivot, then recursively sorts each side — average O(n log n), worst "
            "case O(n^2) on already-sorted input with a poor pivot choice, but it "
            "sorts in-place, making it faster in practice for most datasets."
        ),
    },
    {
        "subject": "data_structures", "title": "Graphs: BFS and DFS",
        "text": (
            "A graph is a set of nodes (vertices) connected by edges. Breadth-First "
            "Search (BFS) explores level by level using a queue — it finds the "
            "shortest path in an unweighted graph. Depth-First Search (DFS) explores "
            "as far as possible down one path before backtracking, typically using a "
            "stack or recursion — useful for detecting cycles, topological sorting, "
            "and connectivity checks."
        ),
    },

    # ---------------- Databases ----------------
    {
        "subject": "databases", "title": "Relational Databases and SQL Basics",
        "text": (
            "A relational database organizes data into tables (rows and columns) "
            "connected by relationships. SQL (Structured Query Language) is used to "
            "define and manipulate this data: SELECT retrieves rows, INSERT adds "
            "rows, UPDATE modifies rows, DELETE removes rows, and WHERE filters "
            "which rows are affected. Example: SELECT name FROM students WHERE "
            "grade > 90;"
        ),
    },
    {
        "subject": "databases", "title": "Primary Keys, Foreign Keys, and Joins",
        "text": (
            "A primary key uniquely identifies each row in a table (e.g., "
            "student_id). A foreign key is a column in one table that references "
            "the primary key of another table, enforcing referential integrity "
            "between related tables. A JOIN combines rows from two tables based on "
            "a related column: INNER JOIN returns only matching rows, LEFT JOIN "
            "returns all rows from the left table plus matches from the right (NULL "
            "where there's no match)."
        ),
    },
    {
        "subject": "databases", "title": "Normalization",
        "text": (
            "Normalization organizes a database to reduce redundancy and avoid "
            "update anomalies, done in stages called normal forms. 1NF requires "
            "atomic column values (no repeating groups). 2NF removes partial "
            "dependencies on a composite key. 3NF removes transitive dependencies "
            "(non-key columns depending on other non-key columns). Highly "
            "normalized schemas save storage and keep data consistent, at the cost "
            "of needing more JOINs to reconstruct information."
        ),
    },
    {
        "subject": "databases", "title": "Indexes and Query Optimization",
        "text": (
            "An index is a data structure (often a B-tree) built on one or more "
            "columns that speeds up lookups, similar to an index in a book, at the "
            "cost of extra storage and slower writes (the index must be updated on "
            "every INSERT/UPDATE/DELETE). Query optimization techniques include "
            "indexing columns used in WHERE/JOIN clauses, avoiding SELECT * when "
            "only specific columns are needed, and using EXPLAIN to inspect the "
            "database's query execution plan."
        ),
    },
    {
        "subject": "databases", "title": "ACID Properties and Transactions",
        "text": (
            "A transaction is a sequence of database operations treated as a single "
            "unit of work. ACID describes the guarantees a reliable transaction "
            "system provides: Atomicity (all operations succeed or none do), "
            "Consistency (the database moves between valid states), Isolation "
            "(concurrent transactions don't interfere with each other), and "
            "Durability (once committed, changes survive a crash)."
        ),
    },
    {
        "subject": "databases", "title": "SQL vs. NoSQL",
        "text": (
            "SQL (relational) databases like MySQL and PostgreSQL use fixed schemas "
            "and tables, and excel at complex queries and strong consistency. NoSQL "
            "databases like MongoDB (document-based), Redis (key-value), and "
            "Cassandra (wide-column) offer flexible schemas and horizontal "
            "scalability, trading some consistency guarantees for speed and "
            "scale — a good fit for large, fast-changing, or unstructured data."
        ),
    },

    # ---------------- Machine Learning ----------------
    {
        "subject": "machine_learning", "title": "Supervised vs. Unsupervised Learning",
        "text": (
            "In supervised learning, the model learns from labeled data (input-output "
            "pairs) to predict outputs for new inputs — examples include "
            "classification (predicting a category) and regression (predicting a "
            "number). In unsupervised learning, the model finds patterns in "
            "unlabeled data — examples include clustering (grouping similar points, "
            "e.g., k-means) and dimensionality reduction (e.g., PCA)."
        ),
    },
    {
        "subject": "machine_learning", "title": "Overfitting and Underfitting",
        "text": (
            "Overfitting happens when a model learns the training data too well, "
            "including its noise, and performs poorly on new/unseen data — shown by "
            "high training accuracy but low test accuracy. Underfitting happens when "
            "a model is too simple to capture the underlying pattern, performing "
            "poorly on both training and test data. Fixes for overfitting include "
            "regularization, more training data, dropout (in neural networks), and "
            "cross-validation; fixes for underfitting include a more complex model "
            "or better features."
        ),
    },
    {
        "subject": "machine_learning", "title": "Train/Test Split and Cross-Validation",
        "text": (
            "Data is typically split into a training set (to fit the model) and a "
            "test set (to evaluate it on unseen data), commonly 80/20. "
            "Cross-validation (e.g., k-fold) splits the training data into k parts, "
            "trains on k-1 folds and validates on the remaining fold, repeating k "
            "times and averaging the results — giving a more reliable estimate of "
            "model performance than a single split, especially with limited data."
        ),
    },
    {
        "subject": "machine_learning", "title": "Evaluation Metrics: Precision, Recall, F1",
        "text": (
            "Accuracy is the percentage of correct predictions, but it's misleading "
            "on imbalanced data. Precision is the fraction of predicted positives "
            "that are truly positive (how trustworthy a positive prediction is). "
            "Recall is the fraction of actual positives correctly identified (how "
            "many positives you catch). The F1-score is the harmonic mean of "
            "precision and recall, useful when you need a single balanced metric."
        ),
    },
    {
        "subject": "machine_learning", "title": "Common Algorithms: Logistic Regression, Decision Trees, Random Forests",
        "text": (
            "Logistic regression predicts a probability using a linear combination "
            "of features passed through a sigmoid function — simple, fast, and "
            "interpretable, good as a baseline. A decision tree splits data on "
            "feature thresholds to make predictions, easy to interpret but prone to "
            "overfitting. A random forest trains many decision trees on random "
            "subsets of data/features and averages their predictions, reducing "
            "overfitting and usually outperforming a single tree."
        ),
    },
    {
        "subject": "machine_learning", "title": "Neural Networks and Deep Learning Basics",
        "text": (
            "A neural network consists of layers of connected nodes (neurons). Each "
            "connection has a weight, and each neuron applies an activation "
            "function (like ReLU or sigmoid) to introduce non-linearity. During "
            "training, backpropagation computes the gradient of the loss function "
            "with respect to each weight, and gradient descent updates the weights "
            "to reduce the loss. 'Deep' learning simply means using many layers, "
            "which lets the network learn increasingly abstract features."
        ),
    },

    # ---------------- Statistics ----------------
    {
        "subject": "statistics", "title": "Mean, Median, and Standard Deviation",
        "text": (
            "The mean is the average of all values (sum divided by count). The "
            "median is the middle value when data is sorted, which is more robust "
            "to outliers than the mean. Standard deviation measures how spread out "
            "values are around the mean — a small standard deviation means values "
            "cluster tightly around the mean, a large one means they're spread out."
        ),
    },
    {
        "subject": "statistics", "title": "Probability Distributions",
        "text": (
            "A probability distribution describes how likely different outcomes are. "
            "The normal (Gaussian) distribution is a symmetric bell curve common in "
            "natural phenomena, defined by its mean and standard deviation. Other "
            "common distributions include the binomial (number of successes in n "
            "trials) and the Poisson (count of events in a fixed interval)."
        ),
    },
    {
        "subject": "statistics", "title": "Hypothesis Testing and p-values",
        "text": (
            "Hypothesis testing checks whether an observed effect is likely real or "
            "due to chance. You define a null hypothesis (no effect) and an "
            "alternative hypothesis (an effect exists), then compute a p-value — "
            "the probability of observing your data (or more extreme) if the null "
            "hypothesis were true. A p-value below a chosen threshold (commonly "
            "0.05) leads to rejecting the null hypothesis."
        ),
    },
    {
        "subject": "statistics", "title": "Correlation vs. Causation",
        "text": (
            "Correlation measures how strongly two variables move together, ranging "
            "from -1 (perfect negative) to +1 (perfect positive), with 0 meaning no "
            "linear relationship. Correlation does not imply causation — two "
            "variables can be correlated because one causes the other, both are "
            "caused by a third variable (a confounder), or purely by coincidence."
        ),
    },
    {
        "subject": "statistics", "title": "The Central Limit Theorem",
        "text": (
            "The Central Limit Theorem states that if you repeatedly take "
            "sufficiently large random samples from any population and compute "
            "their means, the distribution of those sample means will approximate "
            "a normal distribution, regardless of the population's original "
            "distribution shape. This is why the normal distribution is so central "
            "to statistical inference and confidence intervals."
        ),
    },
]

out_dir = os.path.dirname(os.path.abspath(__file__))
kb_path = os.path.join(out_dir, "knowledge_base.json")

# Assign stable chunk ids
for i, chunk in enumerate(KB):
    chunk["chunk_id"] = f"kb_{i:03d}"

with open(kb_path, "w", encoding="utf-8") as f:
    json.dump(KB, f, indent=2, ensure_ascii=False)

print(f"Knowledge base saved to: {kb_path}")
print(f"Total chunks: {len(KB)}")
from collections import Counter
print(Counter(c["subject"] for c in KB))
