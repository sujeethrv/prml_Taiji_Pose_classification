import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import MLP, ResNet, CombinedModel
from datasets_loaders import TaijiDataset
import torch.optim as optim
from torch.autograd import Variable
from utils import plot_confusion_matrix, get_metrics, save_predictions
import torch.nn as nn
import json
from pathlib import Path

mocap_input_dim = 51
footp_input_dim = 2520
num_epochs = 10
learning_rate = 1e-4
num_classes = 46
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_root_dir = './dataset_100_per_pose'
for test_subj in range(1,11):
    test_subject = 'Subject'+str(test_subj)
    results_root_dir = "./runs/ten_subjs_run8_10_epochs/"+test_subject
    Path(results_root_dir).mkdir(parents=True, exist_ok=True)

    train_dataset = TaijiDataset(dataset_root_dir, transform=transform, test=False, test_subject=test_subject)
    test_dataset = TaijiDataset(dataset_root_dir, transform=transform, test=True, test_subject=test_subject)
    print("len(train_dataset) :",len(train_dataset))
    print("len(test_dataset) :",len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    mocap_model  = MLP(mocap_input_dim, h1 = 32,   h2 = 16,  output_dim = 8)
    footp_model  = MLP(footp_input_dim, h1 = 1024, h2 = 512, output_dim = 256)
    resnet_model = ResNet()

    # Hyperparameters and settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if mocap_model==None and footp_model==None:
        print("Resnet only")
        combined_model = CombinedModel(resnet_model, mocap_model, 0, footp_model, 0, num_classes)
    else:
        print("All 3 models")
        combined_model = CombinedModel(resnet_model, mocap_model, mocap_model.layers[-1].out_features, footp_model, footp_model.layers[-1].out_features, num_classes)
    combined_model.to(device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        combined_model.train()
        train_loss = 0.0
        train_true_labels = []
        train_predicted_labels = []

        for batch_idx, (image_input, mocap_input, footp_input, labels) in enumerate(train_loader):
            image_input = image_input.to(device)
            mocap_input = mocap_input.to(device)
            footp_input = footp_input.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = combined_model(image_input, mocap_input, footp_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_true_labels.extend(labels.detach().cpu().numpy())
            train_predicted_labels.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}")

        # Testing loop
        combined_model.eval()
        test_true_labels = []
        test_predicted_labels = []

        with torch.no_grad():
            for image_input, mocap_input, footp_input, labels in test_loader:
                image_input = image_input.to(device)
                mocap_input = mocap_input.to(device)
                footp_input = footp_input.to(device)
                labels = labels.to(device)

                outputs = combined_model(image_input, mocap_input, footp_input)
                test_true_labels.extend(labels.detach().cpu().numpy())
                test_predicted_labels.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())

    class_names = [str(i) for i in range(46)]

    # Save the model
    torch.save(combined_model.state_dict(), results_root_dir+"/combined_model.pth")

    # Calculate and save performance metrics
    get_metrics(train_true_labels, train_predicted_labels, class_names, results_root_dir+"/train_classification_report.csv")
    get_metrics(test_true_labels, test_predicted_labels, class_names, results_root_dir+"/test_classification_report.csv")

    plot_confusion_matrix(train_true_labels, train_predicted_labels, class_names, results_root_dir+"/train_confusion_matrix.png")
    plot_confusion_matrix(test_true_labels, test_predicted_labels, class_names, results_root_dir+"/test_confusion_matrix.png")

    save_predictions(train_true_labels, train_predicted_labels, results_root_dir+"/train_predictions.csv")
    save_predictions(test_true_labels, test_predicted_labels, results_root_dir+"/test_predictions.csv")

    params = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'optimizer': type(optimizer).__name__,
        'criterion': type(criterion).__name__,
        'device': device.type,
        'resnet18_pretrained': True,
        'mlp_mocap': {
            'input_dim': mocap_input_dim,
            'h1': 32,
            'h2': 16,
            'output_dim': 8
        },
        'mlp_footp': {
            'input_dim': footp_input_dim,
            'h1': 1024,
            'h2': 512,
            'output_dim': 256
        },
        'combined_model': {
            'fc1_output': 256,
            'fc2_output': 128,
            'num_classes': num_classes
        }
    }

    with open(results_root_dir+"/training_params.json", 'w') as f:
        json.dump(params, f, indent=4)

    print(f"Training and testing complete for test_subject {test_subj}. Saved performance metrics and predictions.")