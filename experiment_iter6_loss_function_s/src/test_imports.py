#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(line_buffering=True)
print("1: starting", flush=True)
import numpy, pandas
print("2: numpy/pandas", flush=True)
import torch
print(f"3: torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, LayerNorm
print("4: pyg OK", flush=True)
print("DONE", flush=True)
