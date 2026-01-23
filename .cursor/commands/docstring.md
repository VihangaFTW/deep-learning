# docstring

# Docstring Rules (Python)

## Goal
Add clear, concise docstrings that explain **intent and non-obvious behavior** without verbosity or boilerplate.

---

## General Rules
- Follow **PEP 257** conventions.
- Prefer clarity over completeness.
- Do not restate obvious code or type hints.
- Avoid boilerplate sections unless they add real value.

---

## Existing Docstrings
- Keep existing docstrings if they are accurate and clear.
- Refactor only if outdated, misleading, redundant, or verbose.
- Do not rewrite solely for stylistic consistency.

---

## Module-Level Docstrings
- Required for every Python file.
- Briefly describe the module’s purpose or responsibility.
- Avoid implementation details, metadata, or usage examples.

---

## Function Docstrings
- Required for all functions.
- Describe what the function does and any non-obvious behavior or side effects.
- One-line docstrings are sufficient for self-explanatory functions.
- Document parameters and returns only when intent is not obvious.

---

## Class Docstrings
- Required for all classes.
- Describe what the class represents and its responsibility.
- Do not list attributes unless clarification is necessary.

---

## Method Docstrings
- Required for all methods.
- Follow the same rules as function docstrings.
- Keep `__init__` docstrings minimal; document parameters only if non-obvious.
- Private methods may use very concise docstrings or omit them if intent is clear.

---

## Style Constraints
- Use imperative mood (e.g., “Return parsed tokens.”).
- Keep one-line docstrings under ~80 characters where practical.
- Use multi-line docstrings only when necessary.
- Avoid structured sections (`Args`, `Returns`, `Raises`) unless they improve clarity.

---

## Guiding Principle
> Docstrings explain **why and intent**, not what is already obvious from the code.
