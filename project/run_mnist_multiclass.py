import time

from mnist import MNIST

import minitorch

mndata = MNIST("project/data/")
images, labels = mndata.load_training()

BACKEND = minitorch.TensorBackend(minitorch.FastOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        # TODO: Implement for Task 4.5.
        return minitorch.conv2d(input, self.weights.value) + self.bias.value
        # raise NotImplementedError("Need to implement for Task 4.5")


class Network(minitorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self):
        super().__init__()

        # For vis
        self.mid = None
        self.out = None

        # TODO: Implement for Task 4.5.
        self.conv1 = Conv2d(in_channels=1, out_channels=4, kh=3, kw=3)
        self.conv2 = Conv2d(in_channels=4, out_channels=8, kh=3, kw=3)

        # Fully connected layers
        self.fc1 = Linear(
            392, 64
        )  # 392 is derived from the output shape of the pooling layer
        self.fc2 = Linear(64, C)

        # raise NotImplementedError("Need to implement for Task 4.5")

    def forward(self, x):
        # TODO: Implement for Task 4.5.
        # Apply first convolution and ReLU
        self.mid = (self.conv1(x)).relu()

        # Apply second convolution and ReLU
        self.out = (self.conv2(self.mid)).relu()

        # Apply pooling
        pooled = minitorch.maxpool2d(self.out, (4, 4))

        # Flatten the tensor for the linear layer
        # print(pooled._tensor.shape)
        flattened = pooled.view(BATCH, 392)  # Reshape to [batch_size, feature_size]
        # print(flattened._tensor.shape)
        # print(flattened)
        # Apply first Linear layer, ReLU and Dropout
        fc1_out = minitorch.dropout((self.fc1(flattened)).relu(), 0.25)

        # Apply second Linear layer
        fc2_out = self.fc2(fc1_out)

        # Apply logsoftmax over the class dimension
        return minitorch.logsoftmax(fc2_out, dim=1)
        # raise NotImplementedError("Need to implement for Task 4.5")


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


def log_time(epoch, time_per_epoch):
    print(
        f"Time per epoch (excludes validation time): {time_per_epoch} secs on Epoch={epoch}"
    )


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=BACKEND))

    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        # print(n_training_samples)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        acc_time = 0
        for epoch in range(1, max_epochs + 1):
            print(epoch)
            total_loss = 0.0

            model.train()

            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):
                batch_start = time.time()
                if n_training_samples - example_num <= BATCH:
                    continue
                y = minitorch.tensor(
                    y_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()
                acc_time += time.time() - batch_start
                if batch_num % 5 == 0 and epoch % 10 == 0:
                    print(time.time() - batch_start)
                    model.eval()
                    # Evaluate on 5 held-out batches
                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(
                            y_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        x = minitorch.tensor(
                            X_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model)

                    total_loss = 0.0
                    model.train()
                    # print(time.time() - batch_start)
            if epoch % 10 == 0:
                log_time(epoch, acc_time / 10)
                acc_time = 0

    """
    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            start = time.time()
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):
                batch_start = time.time()
                if n_training_samples - example_num <= BATCH:
                    continue
                y = minitorch.tensor(
                    y_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()

                if batch_num % 5 == 0 and epoch % 10 == 0:
                    end = time.time()
                    model.eval()
                    # Evaluate on 5 held-out batches

                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(
                            y_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        x = minitorch.tensor(
                            X_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model, end-start)

                    total_loss = 0.0
                    model.train()
    """


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.01)
