# utils.py

def say_hello(name: str) -> str:
    """Return a hello message for the given name."""
    return f"Hello, {name}!"

def add_numbers(a: int, b: int) -> int:
    """Return the sum of two numbers."""
    return a + b

def pipeline() -> None:
    """Example pipeline with no external input."""
    msg = say_hello("World")
    total = add_numbers(3, 4)
    print(msg)
    print(f"3 + 4 = {total}")
