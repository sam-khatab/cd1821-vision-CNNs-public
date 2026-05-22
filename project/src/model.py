import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.net = nn.Sequential(
        # convolutional layer (sees 3x224x224 image tensor)
        nn.Conv2d(3, 16, 3, padding=1),
        #ReLU activation function (sees 16x224x224 tensor)
        nn.ReLU(),
        # max pooling layer
        nn.MaxPool2d(2, 2),
        # convolutional layer (sees 16x112x112 tensor)
        nn.Conv2d(16, 32, 3, padding=1),
        #ReLU activation function (sees 32x112x112 tensor)
        nn.ReLU(),
        # max pooling layer
        nn.MaxPool2d(2, 2),
        #Flatten layer (32x56x56 -> 100352)
        nn.Flatten(),
        # linear layer (32 * 56 * 56 -> 500)
        nn.Linear(32 * 56 * 56, 500),
        #ReLU activation function (sees 500 tensor)
        nn.ReLU(),
        #Dropout layer
        nn.Dropout(p=dropout),
        # linear layer (500 -> num_classes)
        nn.Linear(500, num_classes) #logits as output so no softmax at end of the network
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.net(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
