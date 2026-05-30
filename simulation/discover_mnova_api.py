"""
discover_mnova_api.py
---------------------
Run this inside MNova to find the correct Python API module name and methods.

    SPINHANCE_XML_DIR=/tmp/x SPINHANCE_OUT_DIR=/tmp/y \
    /Applications/MestReNova.app/Contents/MacOS/MestReNova \
      --nogui \
      --py /abs/path/to/simulation/discover_mnova_api.py
"""

import sys
import os

print("=" * 60)
print("MNova Python API Discovery")
print("=" * 60)

# ── 1. sys.path ───────────────────────────────────────────────────────────────
print("\n--- sys.path ---")
for p in sys.path:
    print(" ", p)

# ── 2. Pre-loaded modules (the MNova API is usually already injected) ─────────
print("\n--- Pre-loaded modules (non-stdlib, non-underscore) ---")
interesting = [
    k for k in sys.modules
    if not k.startswith("_")
    and "." not in k
    and k not in ("sys", "os", "builtins", "site", "abc", "io",
                  "posix", "signal", "errno", "stat", "genericpath",
                  "posixpath", "fnmatch", "re", "sre_compile",
                  "sre_constants", "sre_parse", "copyreg", "types",
                  "warnings", "importlib", "encodings", "codecs",
                  "functools", "operator", "collections", "itertools",
                  "keyword", "heapq", "reprlib", "weakref", "string",
                  "struct", "threading", "time", "enum", "zipimport",
                  "zipfile", "tokenize", "token", "linecache",
                  "traceback", "contextlib", "inspect")
]
for k in sorted(interesting):
    print(" ", k)

# ── 3. Try common MNova module names ─────────────────────────────────────────
print("\n--- Trying known module names ---")
candidates = [
    "Mnova", "mnova", "MNova", "MestReNova", "mestrelab",
    "MestrelabNMR", "NMR", "nmr", "Application", "app",
    "MnovaCore", "SpinSimulation", "Mestrelab",
]
for name in candidates:
    try:
        mod = __import__(name)
        print(f"  FOUND: {name}  →  {mod}")
        print(f"    dir: {[x for x in dir(mod) if not x.startswith('_')]}")
    except ImportError:
        print(f"  not found: {name}")

# ── 4. Check globals for injected objects ─────────────────────────────────────
print("\n--- Globals that look like MNova objects ---")
g = globals().copy()
g.update(vars(sys.modules.get("builtins", sys.modules.get("__builtin__", {}))))
for k, v in sorted(g.items()):
    if k.startswith("_"):
        continue
    t = type(v).__name__
    if t not in ("module", "type", "function", "builtin_function_or_method",
                 "str", "int", "float", "bool", "list", "dict", "NoneType"):
        print(f"  {k}: {t} = {v}")

# ── 5. Check if 'Application' exists as a builtin ────────────────────────────
print("\n--- Checking builtins ---")
import builtins
mnova_builtins = [x for x in dir(builtins) if not x.startswith("_")
                  and x not in dir(__builtins__.__class__)]
# Just print anything that looks non-standard
non_std = [x for x in dir(builtins)
           if not x.startswith("_")
           and x[0].isupper()
           and x not in ("ArithmeticError", "AssertionError", "AttributeError",
                         "BaseException", "BlockingIOError", "BrokenPipeError",
                         "BufferError", "BytesWarning", "ChildProcessError",
                         "ConnectionAbortedError", "ConnectionError",
                         "ConnectionRefusedError", "ConnectionResetError",
                         "DeprecationWarning", "EOFError", "EnvironmentError",
                         "Exception", "FileExistsError", "FileNotFoundError",
                         "FloatingPointError", "FutureWarning", "GeneratorExit",
                         "IOError", "ImportError", "ImportWarning", "IndentationError",
                         "IndexError", "InterruptedError", "IsADirectoryError",
                         "KeyError", "KeyboardInterrupt", "LookupError",
                         "MemoryError", "ModuleNotFoundError", "NameError",
                         "NotADirectoryError", "NotImplemented", "NotImplementedError",
                         "OSError", "OverflowError", "PendingDeprecationWarning",
                         "PermissionError", "ProcessLookupError", "RecursionError",
                         "ReferenceError", "ResourceWarning", "RuntimeError",
                         "RuntimeWarning", "StopAsyncIteration", "StopIteration",
                         "SyntaxError", "SyntaxWarning", "SystemError",
                         "SystemExit", "TabError", "TimeoutError", "True",
                         "TypeError", "UnboundLocalError", "UnicodeDecodeError",
                         "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError",
                         "UnicodeWarning", "UserWarning", "ValueError", "Warning",
                         "ZeroDivisionError", "False", "None",
                         "Ellipsis", "NotImplemented")]
for x in non_std:
    obj = getattr(builtins, x)
    print(f"  builtins.{x}: {type(obj).__name__} = {obj}")

print("\n" + "=" * 60)
print("Discovery complete.")
print("=" * 60)
