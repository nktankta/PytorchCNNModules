# PyTorch CNN Modules

- [Introduction](#introduction)
- [Features](#features)
- [Modules](#modules)
- [Requirement](#requirement)
- [Install](#install)
- [Usage](#usage)
- [Licenses](#licenses)

## Introduction
Pytorch CNN Modules is a collection of CNN modules like the Inception Module.
These modules can be used in a single line.
There is no need to research or implement anything.

## Features
All modules can be easily changed to residual mode and dense mode.
All modules can add SE module.

## Requirement
torch 1.7.0 or above

## Install
```bash
TODO
```

## Usage
create bottleneck module
```python
from PytorchCNNModules import *
module = BottleneckResidual(in_feature, out_feature).to_residual()

output = module(input)
```

## License
Pytorch CNN Modules is provided as open source under the MIT License, see LICENSE.