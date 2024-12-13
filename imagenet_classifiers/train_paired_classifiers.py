import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import random
from datetime import datetime

MODEL_COUNT = 2_000 
SEED = random.choice(range(int(1e10)))
torch.manual_seed(SEED)
random.seed(SEED)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the full ImageNet dataset
imagenet_data = datasets.ImageNet(root='/home/localdia/model_talk/imagenet_classifiers/data/imagenet', split='train', transform=transform)
print(f'Number of images in the full ImageNet dataset: {len(imagenet_data)}')

# Get all class indices
all_class_indices = list(imagenet_data.class_to_idx.values())

# Define a very small neural network
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Print model summary and parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(Classifier())}")
print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'}")

def train_model(class1, class2, success):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128 
    
    selected_indices = []
    class_indices = [class1, class2]
    for class_idx in class_indices:
        selected_indices.extend([i for i, (_, label) in enumerate(imagenet_data.samples) if label == class_idx])

    subset = Subset(imagenet_data, selected_indices)

    idx_to_label = {idx: i for i, idx in enumerate(class_indices)}

    def map_labels(label):
        return idx_to_label[label.item()]

    total_size = len(subset)
    test_size = int(0.10 * total_size)
    val_size = int(0.10 * total_size)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(subset, [train_size, val_size, test_size])

    workers = 12 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    def validate(model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = torch.tensor([map_labels(label) for label in labels]).to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    best_val_acc = 0
    patience = 5
    counter = 0
    best_model_state = None

    for epoch in range(50):  # Max 50 epochs
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.tensor([map_labels(label) for label in labels]).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_acc = validate(model, val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
        
        if val_acc > 0.95 or counter >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_acc = validate(model, test_loader)
    print(f"{datetime.now().isoformat()} {SEED}: model {i+1}/{MODEL_COUNT} for {epoch+1} epochs against classes {class1} and {class2}. Acc: {100 * test_acc:.2f}%. Succ: ~{100 * success:.0f}%")

    return test_acc, model

# Main loop to train models
successes = 1
for i in range(MODEL_COUNT):
    class1, class2 = random.sample(range(1000), 2)
    
    test_acc, model = train_model(class1, class2, successes / (i + 2))
    
    if test_acc >= 0.85:
        filename = f"data/paired_classifiers/model_seed{SEED}_class{class1}vs{class2}_acc{test_acc:.3f}.pth"
        torch.save(model.state_dict(), filename)
        successes = successes + 1

print("Training of all models completed!")