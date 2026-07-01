# Python Bytecode (.pyc) and Script Access Resolution Report

This report explains the occurrence of compiled Python bytecode (`.pyc`) files in the root folder, why they were unreadable, and how the issue has been resolved by restoring the original plain-text Python scripts from the Git history.

---

## 1. Why did the `.pyc` files appear in the root directory?

In Python 3, when you import a module, the interpreter automatically compiles the `.py` source code into compiled bytecode (`.pyc`) files and caches them under a subdirectory called `__pycache__` to speed up subsequent imports.

In the latest commit (`ffb7609cfb75274dcc52d81422f2a3d2bffd92be`), the codebase was transitioned from LSTM/Deep Learning forecasting to statistical nowcasting. As part of this transition, the developer deleted the original `.py` source files for the LSTM model:
*   `architecture.py`
*   `models.py`
*   `sliding_window.py`

However, the Jupyter Notebook [analysis.ipynb](../analysis.ipynb) was not updated and still attempted to import these deleted scripts. 

When execution was triggered in the notebook, Python threw `ModuleNotFoundError` because PEP 3147 prevents Python from importing compiled bytecode from `__pycache__` if the corresponding source `.py` files are missing. To bypass this import issue and restore execution of the notebook cells without changing any logic, the cached `.pyc` files were temporarily copied from `__pycache__` directly to the project root directory, which allowed Python to load them directly.

---

## 2. Why were you unable to access or read them?

`.pyc` files contain **compiled bytecode**, which is a binary representation of the source code intended for the Python virtual machine to execute, rather than plain text for human reading. 
Because the source `.py` files had been deleted, opening the `.pyc` files in a text editor showed unreadable binary gibberish.

---

## 3. How has the issue been resolved?

To restore your access to the source code while keeping the codebase logic completely unchanged, the original plain-text source files and their corresponding weights were restored from the parent commit (`ffb7609cfb75274dcc52d81422f2a3d2bffd92be~1`):

1.  **Restored Files**:
    *   [architecture.py](../architecture.py)
    *   [models.py](../models.py)
    *   [sliding_window.py](../sliding_window.py)
    *   [LSTM_IMPROVEMENTS.md](../LSTM_IMPROVEMENTS.md)
    *   `flare_lstm_weights.pth` (LSTM Model Weights)
    *   `nowcast_model.pt` (Alternative Model Weights)
2.  **Cleaned Up Bytecode**:
    *   Deleted the temporary `.pyc` files (`architecture.pyc`, `models.pyc`, `sliding_window.pyc`) from the project root directory.

Python can now import directly from the plain-text `.py` files normally, and you can freely read, access, and edit the source code of your scripts.
