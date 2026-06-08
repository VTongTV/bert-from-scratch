import math


def warmup_lambda(step, warmup_steps):
    return min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0


def linear_decay_lambda(step, total_steps):
    return max(0.0, 1.0 - step / total_steps)


class WarmupLinearScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr_scale = warmup_lambda(self.current_step, self.warmup_steps) * linear_decay_lambda(self.current_step, self.total_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * lr_scale

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]
