"""
Author: Stanislava Poizlova
Matr.Nr.: K12023677
Exercise 5
"""


import os
import numpy as np
import torch
import torchvision
import torch.utils.data
from datasetclass import Images
from utils import plot
from architectures import SimpleCNN
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import tqdm
import random

random.seed(5)


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            targets, inputs, file_names = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get outputs for network
            outputs = model(inputs)

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance

            # Calculate mean mse loss over all samples in dataloader (accumulate mean losses in `loss`)
            loss += (torch.stack([mse(output, target) for output, target in zip(outputs, targets)]).sum()
                     / len(dataloader.dataset))
    return loss


def main(results_path, network_config: dict, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = int(1e5), device: torch.device = torch.device("cpu")):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # Prepare a path to plot to
    plotpath = os.path.join(results_path, 'plots')
    os.makedirs(plotpath, exist_ok=True)

    # Load dataset
    dataset = Images()




    # Split dataset into training, validation, and test set randomly
    trainingset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (3 / 5))))
    validationset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (3 / 5)),
                                                                               int(len(dataset) * (4 / 5))))
    testset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (4 / 5)),
                                                                         len(dataset)))

    #Create data loaders
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=16, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=16, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=0)


    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

    # Create Network
    net = SimpleCNN(**network_config)
    net.to(device)

    # Get mse loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)

    print_stats_at = 1e2  # print status to tensorboard every x updates
    plot_at = 1e4  # plot every x updates
    validate_at = 5e3  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))

    # Train until n_updates update have been reached
    while update < n_updates:
        for data in trainloader:
            # targets=original_array, inputs=stacked_array
            targets, inputs, ids = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            #print(inputs[:1].shape)
            #targets[:1].shape)

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs for network
            outputs = net(inputs)
            #print(outputs[:1].shape)

            # Calculate loss, do backward pass, and update weights
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print current status and score
            # if update % print_stats_at == 0 and update > 0:
                # writer.add_scalar(tag="training/loss",
                                  # scalar_value=loss.cpu(),
                                  # global_step=update)

            # Plot output
            # if update % plot_at == 0:
                # plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     # plotpath, update)

            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=valloader, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))

            update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progess_bar.update()

            # Increment update counter, exit if maximum number of updates is reached
            update += 1
            if update >= n_updates:
                break

    update_progess_bar.close()
    print('Finished Training!')

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    test_loss = evaluate_model(net, dataloader=testloader, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, device=device)
    train_loss = evaluate_model(net, dataloader=trainloader, device=device)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")

    # Write result to file
    with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
        print(f"Scores:", file=fh)
        print(f"test loss: {test_loss}", file=fh)
        print(f"validation loss: {val_loss}", file=fh)
        print(f"training loss: {train_loss}", file=fh)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as fh:
        config = json.load(fh)
    main(**config)
