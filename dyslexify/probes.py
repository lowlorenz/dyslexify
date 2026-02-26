import torch
from dyslexify.cache.cache import ActivationCache
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def linear_probe_accuracy(
    activations: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    device: str,
    num_epochs: int = 100,
    random_state: int = 42,
    lr: float = 0.1,
):
    activations = activations.squeeze(1).to(device)
    labels = labels.to(device)

    # 1. split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=0.2, random_state=random_state
    )

    # 2. train model
    model = nn.Linear(in_features=activations.shape[-1], out_features=num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        # outputs = F.softmax(outputs, dim=1)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

    # 3. evaluate model
    with torch.no_grad():
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        accuracy = (outputs.argmax(dim=1) == y_test).float().mean()

    # 4. return model
    return accuracy.item()
