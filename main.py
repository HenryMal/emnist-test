from brush import Brush
from network import CharacterRecognizer
import tkinter as tk
import torch

root = tk.Tk()
root.resizable(False, False)
app = Brush(root)
root.mainloop()

