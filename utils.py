import torch
from tqdm import tqdm
from torchmetrics.aggregation import MeanMetric

import numpy as np
import matplotlib.pylab as plt




def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, target_weights, epoch=None, device='cpu'):
  model.train()
  loss_train = MeanMetric()
  metric_train = MeanMetric()

  with tqdm(train_loader, unit='batch') as tepoch:
    for image, targets, extras, state, species in tepoch:
      if epoch:
        tepoch.set_description(f'Epoch {epoch}')

        image = image.to(device)
        targets = targets.to(device)
        extras = extras.to(device)      
        state = state.to(device)
        species = species.to(device)

        outputs = model(image, extras, state, species)

        loss = loss_fn(outputs, targets, target_weights)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        r2 = metric(targets, outputs, target_weights)

        loss_train.update(loss.item(), weight=len(targets))
        metric_train.update(r2.item(), weight=len(targets))


        tepoch.set_postfix(loss=loss_train.compute().item(),
                           metric=metric_train.compute().item())

  return model, loss_train.compute().item(), metric_train.compute().item()

def evaluate(model, test_loader, loss_fn, metric, device='cpu'):
  model.eval()
  loss_eval = MeanMetric()
  metric.reset()

  with torch.inference_mode():
    for inputs, targets in test_loader:
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)

      loss = loss_fn(outputs, targets)
      loss_eval.update(loss.item(), weight=len(targets))

      metric(outputs, targets.to(torch.int32))

  return loss_eval.compute().item(), metric.compute().item()


def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load(model, loss, optimizer, device='cpu', reset = False, load_path = None):
    model = model
    loss_fn = loss
    optimizer = optimizer

    if reset == False : 
        if load_path is None :
            print('give path for load model')
        if load_path is not None:
            if device == 'cpu':
                sate = torch.load(load_path,map_location=torch.device('cpu'))
            else :
                sate = torch.load(load_path)
            
            model.load_state_dict(sate['state_dict'])
            # loss_fn.load_state_dict(sate['loss_fun'])
            optimizer.load_state_dict(sate['optimizer'])
            optimizer_to(optimizer, device)
    return model, loss_fn, optimizer
   


def save(save_path, model, optimizer, loss_fn):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'loss_fun' : loss_fn.state_dict()
    }

    torch.save(state, save_path)

def plot(train_hist, valid_hist, label):
    print(f'\nTrained {len(train_hist)} epochs')

    plt.plot(range(len(train_hist)), train_hist, 'k-', label="Train")
    plt.plot(range(len(valid_hist)), valid_hist, 'y-', label="Validation")

    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.grid(True)
    plt.legend()
    plt.show()


def weighted_mse_loss(preds, targets, weights):
    """
    preds, targets: [B, T]
    weights: [T]
    """
    loss = (preds - targets) ** 2
    loss = loss * weights
    return loss.sum(dim=1).mean()


def weighted_r2_score(y_true, y_pred, weights):
    """
    y_true, y_pred: [N, T]
    weights: [T]
    """
    weights = weights.unsqueeze(0)  # [1, T]

    y_bar = (weights * y_true).sum() / weights.sum()

    ss_res = (weights * (y_true - y_pred) ** 2).sum()
    ss_tot = (weights * (y_true - y_bar) ** 2).sum()

    return 1.0 - ss_res / ss_tot



