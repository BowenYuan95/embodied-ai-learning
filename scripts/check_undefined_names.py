#!/usr/bin/env python3
"""Coarse static check for module-level names that are used but never defined.

Why this exists
---------------
``python -m py_compile`` catches syntax only, and ``--help`` exits inside argparse before any
real code runs, so **neither can catch an undefined name**. A script can pass both and still
raise ``NameError`` on its first real invocation. That happened here: a patch that spliced a
function body deleted two module constants and a helper that sat between the spliced region and
``main``, and the module kept passing every check that was run.

This is **not a linter**. It knows about module-level bindings, imports, builtins and the module
dunders, and it collects the names bound anywhere inside each function. That is enough to catch
"used but never defined at module level". It deliberately does not model closures, ``global`` /
``nonlocal``, conditional imports, ``__all__``, or attribute access, so it can report a name as
defined when a stricter tool would not, and it can miss bugs involving nested scopes. Install
``pyflakes`` or ``ruff`` when that matters.

    python scripts/check_undefined_names.py scripts/*.py
"""

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__",
           "__builtins__", "__debug__", "__annotations__"}


def _statement_bindings(node, known: set[str]) -> None:
    """Collect bindings from module-level statements without entering function/class bodies.

    Recursing through statements (not just ``tree.body``) is what picks up loop variables of a
    module-level ``for`` nested inside another statement, e.g. the ``_word`` in

        for _env_id in TASKS:
            for _word in INSTRUCTIONS[_env_id].split():
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        known.add(node.name)
        return
    if isinstance(node, ast.Assign):
        for target in node.targets:
            known.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        known.add(node.target.id)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        known.update((a.asname or a.name).split(".")[0] for a in node.names)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        known.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                known.update(n.id for n in ast.walk(item.optional_vars) if isinstance(n, ast.Name))
    elif isinstance(node, ast.Try):
        for handler in node.handlers:
            if handler.name:
                known.add(handler.name)
    for child in ast.iter_child_nodes(node):
        _statement_bindings(child, known)


def module_bindings(tree: ast.Module) -> set[str]:
    known = set(dir(builtins)) | DUNDERS
    _statement_bindings(tree, known)
    known.update(n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    return known


def function_bindings(fn) -> set[str]:
    args = fn.args
    out = {a.arg for a in [*args.args, *args.kwonlyargs, *args.posonlyargs]}
    for extra in (args.vararg, args.kwarg):
        if extra:
            out.add(extra.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
    return out


def check(path: Path) -> list[tuple[str, str, int]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    known = module_bindings(tree)
    problems: set[tuple[str, str, int]] = set()

    def walk(node, scopes):
        """``scopes`` is a list of ``(function_name, bindings)``, innermost last.

        Walking with an explicit scope chain rather than ``ast.walk`` is what keeps this from
        reporting every closure variable as undefined: without it, a name defined in an
        enclosing function (``env_id`` inside ``load_rgb_dataset``, say) looks undefined from
        the nested function that uses it.
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes = scopes + [(node.name, function_bindings(node))]
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # Comprehension targets are bound only inside the comprehension, so a variable like
            # the `e` in `max(len(...) for e in TASKS)` must not be looked up at module level.
            binds: set[str] = set()
            for gen in node.generators:
                binds |= {n.id for n in ast.walk(gen.target) if isinstance(n, ast.Name)}
            scopes = scopes + [("<comprehension>", binds)]
        visible = set(known)
        for _, bound in scopes:
            visible |= bound
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id not in visible:
                    owner = scopes[-1][0] if scopes else "<module>"
                    problems.add((owner, child.id, child.lineno))
            else:
                walk(child, scopes)

    walk(tree, [])
    return sorted(problems)


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or sorted(Path("scripts").glob("*.py"))
    bad = 0
    for path in targets:
        try:
            problems = check(path)
        except SyntaxError as exc:
            print(f"{path}: SyntaxError {exc}")
            bad += 1
            continue
        for fn, name, line in problems:
            print(f"{path}:{line}: {fn}() uses undefined name {name!r}")
            bad += 1
    print(f"checked {len(targets)} file(s), {bad} undefined name(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
