import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import entropy
from tqdm import tqdm
from data import MolecularGraphDataset
from gin import ContraGIN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


hidden_dim = 30
num_layers = 6
learning_rate = 0.01
num_epochs = 10
batch_size = 32

train_dataset = MolecularGraphDataset('./schneider50k_2014.tsv', split="train")
val_dataset = MolecularGraphDataset('./schneider50k_2014.tsv', split="test")


def collate_fn(batch):
    prec_data, prod_data, y_vals = zip(*batch)
    y_vals = torch.stack([torch.argmax(y) for y in y_vals], dim=0)

    prec_data = [data.to(device) if isinstance(
        data, torch.Tensor) else data for data in prec_data]
    prod_data = [data.to(device) if isinstance(
        data, torch.Tensor) else data for data in prod_data]

    return prec_data, prod_data, y_vals


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, collate_fn=collate_fn)

model = ContraGIN(train_dataset,
                  num_layers, hidden_dim).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    preds = []
    targets = []

    for batch in tqdm(train_loader, desc='Training'):
        prec_data_batch, prod_data_batch, y_true_batch = batch

        y_true_batch = y_true_batch.to(device)

        optimizer.zero_grad()

        prec_outputs, prod_outputs = model(prec_data_batch, prod_data_batch)

        prec_outputs_concat = torch.cat(prec_outputs, dim=0)
        prod_outputs_concat = torch.cat(prod_outputs, dim=0)

        combined_outputs = torch.cat(
            [prec_outputs_concat, prod_outputs_concat], dim=1)

        loss = criterion(combined_outputs, y_true_batch.long())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        preds.extend(torch.argmax(combined_outputs, dim=1).tolist())
        targets.extend(y_true_batch.tolist())

    acc = accuracy_score(targets, preds)
    cen = entropy(targets)
    mcc = matthews_corrcoef(targets, preds)

    return total_loss / len(train_loader), acc, cen, mcc


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            prec_data_batch, prod_data_batch, y_true_batch = batch

            y_true_batch = y_true_batch.to(device)

            prec_outputs, prod_outputs = model(
                prec_data_batch, prod_data_batch)

            prec_outputs_concat = torch.cat(prec_outputs, dim=0)
            prod_outputs_concat = torch.cat(prod_outputs, dim=0)

            combined_outputs = torch.cat(
                [prec_outputs_concat, prod_outputs_concat], dim=1)

            loss = criterion(combined_outputs, y_true_batch.long())
            total_loss += loss.item()

            preds.extend(torch.argmax(combined_outputs, dim=1).tolist())
            targets.extend(y_true_batch.tolist())

    acc = accuracy_score(targets, preds)
    cen = entropy(targets)
    mcc = matthews_corrcoef(targets, preds)

    return total_loss / len(data_loader), acc, cen, mcc


for epoch in range(num_epochs):
    train_loss, train_acc, train_cen, train_mcc = train(
        model, train_loader, optimizer, criterion)
    val_loss, val_acc, val_cen, val_mcc = evaluate(
        model, val_loader, criterion)

    tqdm.write(f'Epoch [{
               epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    tqdm.write(f'Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}')
    tqdm.write(f'Train CEN: {train_cen:.4f}, Val CEN: {val_cen:.4f}')
    tqdm.write(f'Train MCC: {train_mcc:.4f}, Val MCC: {val_mcc:.4f}')
