from pathlib import Path
import torch

path = str(Path(__file__).resolve().with_name("gen"))

state = torch.load(
    path,
    map_location="cpu",
    weights_only=True
)

print("Total:", len(state))

for key, value in state.items():
    print(key, tuple(value.shape))