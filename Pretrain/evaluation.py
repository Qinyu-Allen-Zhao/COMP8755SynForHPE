import torch


def eva_acc_and_loss(data_loader, model, gpu_available):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    # Here we set reduction = 'sum'
    # the loss function will *not* divide the loss of a batch
    # by the batch size.
    # It will be convenient for us to calculate the average
    # loss on the dataset
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    if gpu_available:
        loss_func.cuda()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images, labels = data
            if gpu_available:
                images = images.cuda()
                labels = labels.cuda()

            # Forward
            out = model(images)

            # Count the correct prediction and the total number
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Evaluate the loss
            loss = loss_func(out, labels)
            loss_sum += loss.item()

    acc = 100.0 * correct / total
    loss_avg = loss_sum / total

    return acc, loss_avg
