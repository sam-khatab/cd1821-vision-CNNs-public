import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.3) -> None:

        super().__init__()
        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.net = nn.Sequential(
        
        # convolutional layer (sees 3x224x224 image tensor)
        nn.Conv2d(3, 16, kernel_size = 3, padding=1),
        nn.BatchNorm2d(16),
        #ReLU activation function (sees 16x224x224 tensor)
        nn.ReLU(),
        # max pooling layer
        nn.MaxPool2d(2, 2),

        # convolutional layer (sees 16x112x112 tensor)
        nn.Conv2d(16, 32, kernel_size = 3, padding=1),
        nn.BatchNorm2d(32),
        #ReLU activation function (sees 32x112x112 tensor)
        nn.ReLU(),
        # max pooling layer
        nn.MaxPool2d(2, 2),

         # convolutional layer (sees 32x56x56 tensor)
        nn.Conv2d(32, 64, kernel_size = 3, padding=1),
        nn.BatchNorm2d(64),
        #ReLU activation function (sees 64x56x56 tensor)
        nn.ReLU(),
        # max pooling layer
        nn.MaxPool2d(2, 2),


        #Flatten layer (64x28x28 -> 50176)
        nn.Flatten(),
        # linear layer (64 * 28 * 28 -> 256)
        nn.Linear(64 * 28 * 28, 256),
        #ReLU activation function (sees 256 tensor)
        nn.ReLU(),
        #Dropout layer
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes) #logits as output so no softmax at end of the network
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.net(x)
        return x


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
