import torch
import time
import numpy as np

MAX_TORQUE = 10

model_path = './model/120321_tip_traced.pt'
model = torch.jit.load(model_path)
state = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
action = model(torch.from_numpy(state)).numpy()[0] * MAX_TORQUE
print(action)