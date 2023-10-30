def train(model, device, train_loader, optimizer, epoch, loss_factor, batch_size):
    model.train()
    counter = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        counter += 1
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)

        prediction = output.argmax(1)
        loss = loss_factor(output, targets)
        acc = (output.argmax(1) == targets).sum().item() / batch_size

        loss.backward()
        optimizer.step()

        print(
            f'EPOCH: {epoch + 1}',
            f'STEP: {batch_idx + 1:0>{len(str(len(train_loader)))}}/{len(train_loader)}',
            f'LOSS: {loss.item():.4f}',
            f'ACC: {acc:.4f}',
            f'PREDICTION: {prediction[0]}',
        )
