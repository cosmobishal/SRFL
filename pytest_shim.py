"""Minimal pytest shim so tests run under unittest."""
import unittest

class raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} but nothing was raised")
        return issubclass(exc_type, self.exc_type)

mark = type('mark', (), {'parametrize': staticmethod(lambda *a, **kw: lambda f: f)})()
