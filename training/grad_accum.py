import torch


def gradient_accumulation(model, loss, accumulation_steps, optimizer, scheduler, current_step):
    normalized_loss = loss / accumulation_steps
    normalized_loss.backward()
    if (current_step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
