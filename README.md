[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# SimplifiedTransformers
The author presents an implementation for Simplifying Transformer Blocks. The standard transformer blocks are complex and can lead to architecture instability. In this work, the author investigates how the standard transformer block can be simplified. Through signal propagation theory and empirical observations, the author proposes modifications that remove several components without sacrificing training speed or performance. The simplified transformers achieve the same training speed and performance as standard transformers, while being 15% faster in training throughput and using 15% fewer parameters.


# Install
```
pip3 install --upgrade simplified-transormer-torch

```

--------

## Usage
```python

import torch
from simplified_transformers.main import SimplifiedTransformers

model = SimplifiedTransformers(
    dim=4096,
    depth=6,
    heads=8,
    num_tokens=20000,
)

x = torch.randint(0, 20000, (1, 4096))

out = model(x)
print(out.shape)

```






# License
MIT



