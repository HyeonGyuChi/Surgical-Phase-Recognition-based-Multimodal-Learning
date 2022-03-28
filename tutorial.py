import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray.util.sgd.v2 as sgd
from ray.util.sgd.v2.trainer import Trainer
from ray.tune.integration.mlflow import mlflow_mixin
import mlflow



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # In this example, we don't change the model architecture
        # due to simplicity.
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
    

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

def train(model, optimizer, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

@mlflow_mixin
def train_mnist(config):
    mlflow.autolog()
    
    # Data Setup
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    train_loader = DataLoader(
        datasets.MNIST("~/data", train=True, download=True, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST("~/data", train=False, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for i in range(10):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)

        # Send the current training result back to Tune
        tune.report(mean_accuracy=acc)

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")



######

mlflow.create_experiment("my_experiment")


search_space = {
    "batch_size": tune.choice([1, 2, 4, 8, 16, 32, 64]),
    "epochs": tune.choice([1,2,5,10,20]),
    "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    "momentum": tune.uniform(0.1, 0.9),
    "ray_workers": 2,
    "num_samples": 32,
    "mlflow": {
            "experiment_name": "my_experiment",
            "tracking_uri": mlflow.get_tracking_uri()
        }
}

# Uncomment this to enable distributed execution
# `ray.init(address="auto")`
# {"training_iteration": tune.choice([1, 2, 5, 10, 20, 50, 100])}

# Download the dataset first
datasets.MNIST("~/data", train=True, download=True)

from ray.tune.integration.mlflow import MLflowLoggerCallback

trainer = Trainer(backend="torch",
                    num_workers=search_space["ray_workers"],
                    use_gpu=True)
Trainable = trainer.to_tune_trainable(train_mnist)

analysis = tune.run(train_mnist, 
                    config=search_space,
                    num_samples=search_space["num_samples"],
                    stop={"training_iteration": 100},
                    max_failures=3,
                    scheduler=ASHAScheduler(metric="mean_accuracy", 
                                            time_attr='training_iteration',
                                            max_t=1000,
                                            mode="max"),
                    verbose=2,
                    callbacks=[MLflowLoggerCallback(
                            experiment_name="my_experiment",
                            save_artifact=True)]
                    )

dfs = analysis.trial_dataframes

print(dfs)

# Plot by epoch
ax = None  # This plots everything on the same plot
for d in dfs.values():
    ax = d.mean_accuracy.plot(ax=ax, legend=False)

import matplotlib.pyplot as plt


fig = ax.get_figure()
fig.savefig("output.png")