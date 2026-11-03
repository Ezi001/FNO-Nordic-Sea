from dataclasses import dataclass
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

@dataclass
class CFOTrainArgs:
    num_steps: int = 1000
    random_seed: int = 0
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.99
    do_eval: bool = False
    eval_interval: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    step: int = 0

    def apply_gradients(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return self

def init_cfo_train_state(method, seed, learning_rate, beta1, beta2, device):
    torch.manual_seed(seed)
    model = method.model.to(device)

    x = torch.zeros((1, *method.input_shape), dtype=torch.float32, device=device)
    t = torch.ones(1, dtype=torch.float32, device=device)

    with torch.no_grad():
        if method.use_condition:
            c = torch.ones((1,) + tuple(method.condition_shape), dtype=torch.float32, device=device)
            model(x, t, c)
        else:
            model(x, t)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
    return TrainState(model=model, optimizer=optimizer)

def _prepare_cfo_train_batch(method, raw_batch, generator, device):
    spline_coef = raw_batch["spline_coef"].to(device, non_blocking=True)
    t_start = raw_batch["t_start"].to(device, non_blocking=True)
    t_end = raw_batch["t_end"].to(device, non_blocking=True)

    dt = t_end - t_start
    delta_t = torch.rand(dt.shape, generator=generator, device=device) * dt
    x0 = spline_coef[:, 0]
    eps = torch.randn(x0.shape, generator=generator, device=device)

    return (spline_coef, t_start, t_end, delta_t, eps)

def train_cfo(method, spline_dataloader, args, eval_dataset=None):
    device = torch.device(args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.random_seed)

    state = init_cfo_train_state(
        method,
        seed=args.random_seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        device=device,
    )

    data_iter = iter(spline_dataloader)
    loss_fn = method.loss_fn

    for step in range(args.num_steps):
        raw_batch = next(data_iter)
        batch = _prepare_cfo_train_batch(method, raw_batch, gen, device)

        state.optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(batch)
        loss.backward()
        state.apply_gradients()

        if step % 10 == 0:
            print(f"step={step}, loss={loss.item():.6f}")

    return {"state": state}
