import torch
from torch.export import export
from executorch.exir import to_edge

class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        z = self.softmax(x)
        return z

# 1. torch.export: Defines the program with the ATen operator set.
aten_dialect = export(Softmax(), (torch.ones(2,2)))

# 2. to_edge: Make optimizations for Edge devices
edge_program = to_edge(aten_dialect)

# 3. to_executorch: Convert the graph to an ExecuTorch program
executorch_program = edge_program.to_executorch()

# 4. Save the compiled .pte program
with open("softmax.pte", "wb") as file:
    file.write(executorch_program.buffer)

