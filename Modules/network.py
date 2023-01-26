import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim


class Network(nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden=None):
        layer_one = int(np.sqrt(inputs * outputs))
        if hidden:
            self.hidden_sizes = hidden
        else:
            self.hidden_sizes = [layer_one, int(np.max([layer_one / 2, outputs + 1]))]

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(inputs, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[1], outputs),
            nn.LogSoftmax(dim=1)
        )

        self.criterion = nn.NLLLoss()
        self.loss = None
        self.optimizer = optim.SGD(self.parameters(), lr=0.005, momentum=0.5)
        self.optimizer.zero_grad()

        # storing information

    def train_model(self, x_train, y_train, epochs):
        logps = self.model.forward(x_train)
        self.loss = self.criterion(logps, y_train)

        pbar = tqdm(range(epochs))

        for e in pbar:
            output = self.model.forward(x_train)
            loss = self.criterion(output, y_train)

            # Training pass
            self.optimizer.zero_grad()

            # This is where the model learns by back propagating
            loss.backward()

            # And optimizes its weights here
            self.optimizer.step()

            # Update the progress bar.
            if e % 1000 == 0:
                pbar.set_postfix_str(f'loss: {loss.item():.2f} - accuracy: {self.test_model(x_train, y_train):.2f}')

    def test_model(self, x_test, y_test):
        correct_count, all_count = 0, 0
        for i in range(len(y_test)):
            img = x_test[i].view(1, len(x_test[0]))

            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = self.model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probabilities = list(ps.numpy()[0])
            pred_label = probabilities.index(max(probabilities))
            true_label = y_test.numpy()[i]

            if true_label == pred_label:
                correct_count += 1

            all_count += 1

        return correct_count / all_count

    def get_probability(self, x_test):
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = self.model(x_test.view([1, len(x_test)]))

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])

        return probab