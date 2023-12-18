import matplotlib.pyplot as plt

# # Read loss data from file
# with open('/home/mengy/Othello/losses.txt', 'r') as file:
#     lines = file.readlines()

# with open('/home/mengy/RL/Othello_copy/losses.txt', 'r') as file:
#     lines_test = file.readlines()

# epochs = []
# pi_losses = []
# v_losses = []

# # Parse the data
# i = 0
# for line in lines:
#     if line.startswith('Epoch'):
#         parts = line.split(': ')
#         epoch_part, losses_part = parts[0], parts[1]
#         # epoch = int(epoch_part.split()[1])
#         i += 1
#         # if i <= 20:
#         #     continue
#         pi_loss, v_loss = losses_part.split(',')
#         pi_loss = float(pi_loss.split(' ')[1])
#         v_loss = float(v_loss.split(' ')[2])
#         epochs.append(i)
#         pi_losses.append(pi_loss)
#         v_losses.append(v_loss)

# epochs_test = []
# pi_losses_test = []
# v_losses_test = []

# # Parse the data
# i = 0
# for line in lines_test:
#     if line.startswith('Epoch'):
#         parts = line.split(': ')
#         epoch_part, losses_part = parts[0], parts[1]
#         # epoch = int(epoch_part.split()[1])
#         i += 1
#         # if i <= 20:
#         #     continue
#         pi_loss, v_loss = losses_part.split(',')
#         pi_loss = float(pi_loss.split(' ')[1])
#         v_loss = float(v_loss.split(' ')[2])
#         epochs_test.append(i)
#         pi_losses_test.append(pi_loss)
#         v_losses_test.append(v_loss)


# # Plot the losses
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, pi_losses, label='pi_loss_resnet',linewidth=2)
# plt.plot(epochs, v_losses, label='v_loss_resnet',linewidth=2)
# plt.plot(epochs_test, pi_losses_test, label='pi_loss_addnoise',linewidth=2)
# plt.plot(epochs_test, v_losses_test, label='v_loss_addnoise',linewidth=2)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.legend()
# plt.savefig("loss")
# plt.show()

# Read loss data from file
# with open('/home/mengy/Othello/losses.txt', 'r') as file:
#     lines = file.readlines()

# with open('/home/mengy/RL/Othello_copy/losses.txt', 'r') as file:
#     lines_test = file.readlines()

# Read loss data from file
with open('/home/mengy/Othello/rate.txt', 'r') as file:
    lines = file.readlines()

with open('/home/mengy/RL/Othello_copy/rate.txt', 'r') as file:
    lines_test = file.readlines()

epochs = []
nwins = []
rates = []

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
        epochs.append(i)
        nwins.append(nwin)
        rates.append(rate)

epochs_test = []
nwins_test = []
rates_test = []

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
        epochs_test.append(i)
        nwins_test.append(nwin)
        rates_test.append(rate)


# Plot the losses
plt.figure(figsize=(10, 6))
# plt.plot(epochs, pi_losses, label='pi_loss_resnet',linewidth=2)
plt.plot(epochs, rates, label='rate_resnet',linewidth=2)
# plt.plot(epochs_test, pi_losses_test, label='pi_loss_addnoise',linewidth=2)
plt.plot(epochs_test, rates_test, label='rate_addnoise',linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('rate')
plt.grid(True)
plt.legend()
plt.savefig("loss")
plt.show()