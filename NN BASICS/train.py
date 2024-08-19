import sys
from torch import optim  # Import optimizers like SGD, Adam
from tqdm import tqdm  # Import tqdm for progress bar

from config import get_config  # Import configuration settings
from utils import *  # Import utility functions
from models import *  # Import model definitions
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard for logging

# Uncomment to initialize TensorBoard writer with a specific directory
# writer = SummaryWriter("runs/MNIST/tryingOut_tensorboard")

def main(cfg):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    step = 0  # Initialize step counter for TensorBoard

    # Suitable hyperparameters search to achieve high model performance (using Tensorboard)
    batch_sizes = [32, 500]  # Different batch sizes to experiment with
    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]  # Different learning rates to experiment with
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Class labels for MNIST digits

    # Loop over each batch size
    for batch_size in batch_sizes:
        # Loop over each learning rate
        for learning_rate in learning_rates:
            # Load training and testing data
            train_loader, test_loader = load_data_online(cfg, batch_size)
            model = CNN(cfg.in_channels, cfg.num_classes)  # Initialize the model (e.g., CNN)
            model.to(device)  # Move model to the specified device (GPU/CPU)
            model.train()  # Set the model to training mode
            criterion = nn.CrossEntropyLoss()  # Define loss function (CrossEntropyLoss for classification)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)  # Set optimizer
            writer = SummaryWriter(f"runs/MNIST/MiniBatchSize/ {batch_size} LR {learning_rate}")  # TensorBoard writer

            # Load model checkpoints if needed
            if cfg.load_model:
                load_checkpoints(torch.load("test_model.pth.tar"), model, optimizer)

            # Training loop for each epoch
            for epochs in range(cfg.epochs):
                losses = []  # List to store loss values
                accuracies = []  # List to store accuracy values

                # Save model checkpoint every 2 epochs
                # if epochs % 2 == 0:
                #     checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                #     save_checkpoints(checkpoint)

                # Loop over batches of data
                for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                    data = data.to(device=device)  # Move data to device
                    target = target.to(device=device)  # Move target labels to device

                    # Uncomment to reshape data if needed for NN models
                    # data = data.reshape(data.shape[0], -1)

                    # Forward pass: compute predictions
                    scores = model(data)
                    loss = criterion(scores, target)  # Compute loss
                    losses.append(loss.item())  # Append loss to list

                    # Backward pass: compute gradients and update weights
                    optimizer.zero_grad()  # Zero the gradients
                    loss.backward()  # Backpropagate the loss
                    optimizer.step()  # Update the weights

                    # Calculate running training accuracy
                    _, prediction = scores.max(1)
                    num_correct = (prediction == target).sum()
                    running_train_acc = float(num_correct) / float(data.shape[0])  # Calculate accuracy
                    accuracies.append(running_train_acc)  # Append accuracy to list

                    # TensorBoard logging
                    writer.add_scalar("Training loss", loss, global_step=step)  # Log loss
                    writer.add_scalar("Training accuracy", running_train_acc, global_step=step)  # Log accuracy
                    # Visualize images from the current batch
                    img_grid = torchvision.utils.make_grid(data)  # Create a grid of images
                    writer.add_image("Training Images", img_grid)  # Log images
                    # Visualize weights of the first fully connected layer
                    writer.add_histogram("Weights for the layer fc1", model.fc1.weight)
                    # TensorBoard embedding projector for visualizing high-dimensional data
                    features = data.reshape(data.shape[0], -1)  # Reshape data for embedding
                    class_labels = [classes[label] for label in prediction]  # Get class labels for predictions
                    if batch_idx == 120:  # Log embeddings at specific batch index
                        writer.add_embedding(features, metadata=class_labels, label_img=data, global_step=step)
                    step += step  # Increment step counter for TensorBoard

                # Log hyperparameters and corresponding metrics (accuracy, loss)
                writer.add_hparams({"Learning Rate": learning_rate, "Batch Size": batch_size},
                                   {"Accuracies": sum(accuracies) / len(accuracies),
                                    "Losses": sum(losses) / len(losses)})

                # Print mean loss for the epoch
                mean_loss = sum(losses) / len(losses)
                print(f"\n loss at epoch {epochs} was {mean_loss:.3f}")

            # Uncomment to check accuracy on training and testing sets after training
            # print(f"Accuracy on training set: {check_accuracy(train_loader, model, device) * 100:.2f}")
            # print(f"Accuracy on testing set: {check_accuracy(test_loader, model, device) * 100:.2f}")

# Entry point for the script
if __name__ == '__main__':
    config_args, unparsed_args = get_config()  # Get configuration settings
    main(config_args)  # Call the main function with the config arguments
