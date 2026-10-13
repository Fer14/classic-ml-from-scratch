import ast
import contextlib
import io
import json
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

CELL_SOURCES = [
    ("markdown", "# Tiny analysis\nWe load numbers, summarize them, and fit a line."),
    ("code", "import numpy as np\nrng = np.random.default_rng(0)\nx = np.arange(10)\ny = 3 * x + 2 + rng.normal(0, 0.5, size=10)"),
    ("code", "print('mean of y:', round(float(y.mean()), 3))\ny.shape"),
    ("code", "slope, intercept = np.polyfit(x, y, 1)\nround(float(slope), 3), round(float(intercept), 3)"),
]


def make_notebook(cell_sources):
    cells = []
    for index, (cell_type, source) in enumerate(cell_sources):
        cell = {"id": f"cell-{index}", "cell_type": cell_type, "metadata": {}, "source": source}
        if cell_type == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "cells": cells,
    }


class MiniKernel:
    def __init__(self):
        self.namespace = {}
        self.execution_count = 0

    def run(self, source):
        self.execution_count += 1
        tree = ast.parse(source)
        last_expression = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expression = ast.Expression(tree.body.pop().value)
        captured = io.StringIO()
        result = None
        with contextlib.redirect_stdout(captured):
            exec(compile(tree, "<cell>", "exec"), self.namespace)
            if last_expression is not None:
                result = eval(compile(last_expression, "<cell>", "eval"), self.namespace)
        outputs = []
        if captured.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": captured.getvalue()})
        if result is not None:
            outputs.append({
                "output_type": "execute_result",
                "execution_count": self.execution_count,
                "data": {"text/plain": repr(result)},
                "metadata": {},
            })
        return outputs


def execute_notebook(notebook, kernel):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell["outputs"] = kernel.run(cell["source"])
        cell["execution_count"] = kernel.execution_count
    return notebook


def output_texts(notebook):
    texts = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            texts.extend(output.get("text") or output["data"]["text/plain"] for output in cell["outputs"])
    return texts


def show_outputs(notebook, title):
    print(title)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for output in cell["outputs"]:
            if output["output_type"] == "stream":
                text = output["text"].strip()
            else:
                text = output["data"]["text/plain"]
            print(f"  [{cell['execution_count']}] {output['output_type']:<15} {text}")


print("Step 1: a notebook is just JSON")
notebook = make_notebook(CELL_SOURCES)
serialized = json.dumps(notebook, indent=1)
print(f"  {len(notebook['cells'])} cells, {len(serialized)} characters of JSON")
print("  first lines:")
for line in serialized.splitlines()[:8]:
    print(f"    {line}")

print()
print("Step 2: a kernel is a long-lived namespace")
executed = execute_notebook(make_notebook(CELL_SOURCES), MiniKernel())
show_outputs(executed, "  outputs from MiniKernel:")

print()
print("Step 3: hidden state from out-of-order execution")
kernel = MiniKernel()
kernel.run("threshold = 10")
kernel.run("count = sum(1 for value in range(20) if value > threshold)")
kernel.run("threshold = 15")
stale_count = kernel.run("count")[0]["data"]["text/plain"]
fresh_count = MiniKernel().run("threshold = 15\nsum(1 for value in range(20) if value > threshold)")[0]["data"]["text/plain"]
print(f"  cell order 1,2,3 then read count -> {stale_count} (computed with the OLD threshold)")
print(f"  restart and run all with threshold 15 -> {fresh_count}")
print(f"  variables alive in the kernel: {sorted(name for name in kernel.namespace if not name.startswith('__'))}")

print()
print("Use It: nbformat validates, nbclient runs a real Jupyter kernel")
real_notebook = nbformat.from_dict(make_notebook(CELL_SOURCES))
nbformat.validate(real_notebook)
print("  nbformat.validate: notebook is valid v4")
with tempfile.TemporaryDirectory() as folder:
    notebook_path = Path(folder) / "tiny_analysis.ipynb"
    kernel_manager = KernelManager(kernel_name="python3", transport="ipc", ip=str(Path(folder) / "kernel"))
    NotebookClient(real_notebook, km=kernel_manager, timeout=60).execute()
    kernel_manager.shutdown_kernel()
    nbformat.write(real_notebook, notebook_path)
    reloaded = nbformat.read(notebook_path, as_version=4)
    show_outputs(reloaded, f"  outputs from a real ipykernel (saved and reloaded {notebook_path.name}):")
print(f"  MiniKernel outputs identical to ipykernel outputs: {output_texts(executed) == output_texts(reloaded)}")
