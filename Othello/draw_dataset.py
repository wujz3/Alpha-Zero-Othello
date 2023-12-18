import matplotlib.pyplot as plt

# Read loss data from file
with open('/home/mengy/Othello/rate.txt', 'r') as file:
    lines = file.readlines()

with open('/home/mengy/RL/Othello_copy/rate.txt', 'r') as file:
    lines_test = file.readlines()

epochs = []
nwins = []
rates = []
dataset = []
# Parse the data
i = 0
for line in lines:
    if line.startswith('numIters'):
        parts = line.split(': ')
        epoch_part, losses_part = parts[0], parts[1]
        # epoch = int(epoch_part.split()[1])
        i += 1
        # if i <= 20:
        #     continue
        nwin, rate ,x= losses_part.split(',')
        nwin = float(nwin.split(' ')[1])
        rate = float(rate.split(' ')[2])
        x = float(x.split(' ')[2])
        epochs.append(i)
        nwins.append(nwin)
        rates.append(rate)
        dataset.append(x)

epochs_test = []
nwins_test = []
rates_test = []
dataset_test = []
# Parse the data
i = 0
for line in lines_test:
    if line.startswith('numIters'):
        parts = line.split(': ')
        epoch_part, losses_part = parts[0], parts[1]
        # epoch = int(epoch_part.split()[1])
        i += 1
        # if i <= 20:
        #     continue
        nwin, rate ,x= losses_part.split(',')
        nwin = float(nwin.split(' ')[1])
        rate = float(rate.split(' ')[2])
        x = float(x.split(' ')[2])
        epochs_test.append(i)
        nwins_test.append(nwin)
        rates_test.append(rate)
        dataset_test.append(x)

# Plot the losses
plt.figure(figsize=(10, 6))
# plt.plot(epochs, pi_losses, label='pi_loss_resnet',linewidth=2)
plt.plot(epochs, dataset, label='dataset_resnet',linewidth=2)
# plt.plot(epochs_test, pi_losses_test, label='pi_loss_addnoise',linewidth=2)
plt.plot(epochs_test, dataset_test, label='dataset_addnoise',linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('DatasetNum')
plt.grid(True)
plt.legend()
plt.savefig("dataset")
plt.show()