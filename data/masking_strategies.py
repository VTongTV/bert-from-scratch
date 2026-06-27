import torch


class MaskingStrategy:
    def __init__(self, mask_prob=0.15, replace_mask=0.8, replace_random=0.1, keep_original=0.1):
        self.mask_prob = mask_prob
        self.replace_mask = replace_mask
        self.replace_random = replace_random
        self.keep_original = keep_original

    def apply(self, input_ids, V):
        labels = input_ids.clone()
        mask = torch.rand(input_ids.shape) < self.mask_prob
        labels[~mask] = -100
        masked = input_ids.clone()
        rand_vals = torch.rand(input_ids.shape)
        mask_token = mask & (rand_vals < self.replace_mask)
        random_token = mask & (rand_vals >= self.replace_mask) & (rand_vals < self.replace_mask + self.replace_random)
        masked[mask_token] = 103
        num_random = random_token.sum().item()
        if num_random > 0:
            masked[random_token] = torch.randint(0, V, (int(num_random),))
        return masked, labels, mask.float()


STRATEGIES = {
    "default": MaskingStrategy(0.15, 0.8, 0.1, 0.1),
    "all_mask": MaskingStrategy(0.15, 1.0, 0.0, 0.0),
    "no_random": MaskingStrategy(0.15, 0.8, 0.0, 0.2),
    "no_mask": MaskingStrategy(0.15, 0.0, 0.2, 0.8),
    "all_random": MaskingStrategy(0.15, 0.0, 1.0, 0.0),
    "all_keep": MaskingStrategy(0.15, 0.0, 0.0, 1.0),
}
