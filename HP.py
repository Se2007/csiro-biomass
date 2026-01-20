import torch
from torch import nn
from torch import optim

from dataset import CsiroDataset, Csiro
from torchvision import transforms


from utils import train_one_epoch, weighted_mse_loss, weighted_r2_score
from ModelSreuctures import BiomassModel


from prettytable import PrettyTable
from colorama import Fore, Style, init


#########################################################################################
## Function for load the model for change and find the hyperparameter during the training
#########################################################################################


def load(model, device='cpu', reset = False, load_path = None):
    model = model

    if reset == False : 
        if load_path is None :
            print('give path for load model')
        if load_path is not None:
            if device == 'cpu':
                sate = torch.load(load_path,map_location=torch.device('cpu'))
            else :
                sate = torch.load(load_path)
            
            model.load_state_dict(sate['state_dict'])
    return model
#####################
####  Arguments #####
#####################


device = 'cuda'
num_epochs = 5
reset = True

Transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(
        mean=[0.4417, 0.5036, 0.3057],
        std=[0.1771, 0.1744, 0.1681])
    ]) 



target_weights = torch.tensor(
    [0.1, 0.1, 0.1, 0.5, 0.2],
    device=device
)

#######################
#   Load DataLoader  #  
#######################

train_loader = Csiro(
                    root="./csiro-biomass",
                    image_root="./csiro-biomass/",
                    transform=Transform,
                    valid_ratio=0.2,
                    seed=42,
                    mini=True   
                    )(batch_size=10)

load_path = ''

#######################
#   Hyperparameters   #
#######################

learning_rates = [0.01,0.03, 0.001, 0.003,  0.0001, 0.0003]
weight_decays = [1e-2, 1e-4, 1e-6, 1e-8]

## preprocessing for makeing the table 

loss_list = []

best_lr = None
best_wd = None
best_loss = float('inf')  
min_num = float('inf')
second_min = float('inf')

table = PrettyTable()
table.field_names = ["LR \ WD"] + [f"WD {i}" for i in weight_decays]

## Loss function and Metric

metric = weighted_r2_score#.to(device)

loss_fn = weighted_mse_loss


for lr in learning_rates:
    for wd in weight_decays:
    
        print(f'\nLR={lr}, WD={wd}')

        ## Model and Optimizer
        

        model = BiomassModel(num_states=4, num_species=15, num_targets=5).to(device)

        ### Calculate the amount of parameters
        print(sum(p.numel() for p in model.parameters()))
        
        model = load(model, device=device, reset = reset, load_path = load_path)
        
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=False)


        for epoch in range(1, num_epochs+1):
            model, loss, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, metric, target_weights, epoch, device=device)

     
        loss_list.append(float(f'{loss:.4f}'))

## Add the color to the first and second minimun loss of the table

sorted_list = sorted(loss_list)
first_min = sorted_list[0]
second_min = sorted_list[1]

first_min_idx = loss_list.index(first_min)
second_min_idx = loss_list.index(second_min)

loss_list[first_min_idx] = f"{Fore.GREEN}{first_min}{Fore.WHITE}"
loss_list[second_min_idx] = f"{Fore.YELLOW}{second_min}{Fore.WHITE}"
loss_list = list(map(str, loss_list))

## Making the table

o = 0

for i in learning_rates:
    row = [f"LR {i}"]

    losses = loss_list[o:len(weight_decays)+o]
    o += len(weight_decays)

    row += losses
    table.add_row(row)


print(table)