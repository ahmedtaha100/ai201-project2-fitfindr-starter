"""
Root conftest.py — present so `pytest tests/` resolves imports of the
top-level modules (tools, agent, utils) from the repo root regardless of
the directory pytest is invoked from.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
