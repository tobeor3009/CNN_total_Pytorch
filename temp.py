from torch.utils.tensorboard import SummaryWriter
ckpt_dir = '/ckpts'
log_dir = './logs'

os.makedirs(log_dir, exist_ok=True)

start_epoch = 0
lr = 1e-4
optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))


def loss_fun(logit, y):
    logit_age = logit[..., 0]
    y_age = y[0]
    logit_gender = logit[..., 1]
    y_gender = y[1].float()
    age_loss = nn.L1Loss().cuda()(logit_age, y_age)
    gender_loss = nn.BCELoss().cuda()(logit_gender, y_gender)

    return age_loss + gender_loss


num_epoch = 100

num_train = len(train_dataset)
num_train_for_epoch = np.ceil(num_train / batch_size)
num_val = len(valid_dataset)
num_val_for_epoch = np.ceil(num_val / batch_size)

writer_train = SummaryWriter(log_dir=f"{log_dir}/train")
writer_val = SummaryWriter(log_dir=f"{log_dir}/val")

for epoch in range(start_epoch + 1, num_epoch + 1):

    net.train()
    loss_arr = []

    for batch, (x, y) in enumerate(train_dataset, 1):
        x = x.cuda()
        y = y.cuda()

        pred = net(x)

        optim.zero_grad()

        loss_value = loss(pred, y)
        loss_arr += [loss_value.item()]

        loss_value.backward()
        optim.step()

        print('train : epoch %04d / %04d | Batch %04d \ %04d | Loss %04f | '
              % (epoch, num_epoch, batch, num_train_for_epoch, np.mean(loss_arr)))

        x = fn_tonumpy(x)
        output = fn_tonumpy(pred)

        writer_train.add_image(
            'input', x, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image(
            'output', output, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    # validation
    with torch.no_grad():
        net.eval()
        loss_arr = []
        for batch, (x, y) in enumerate(valid_dataset, 1):
        x = x.cuda()
        y = y.cuda()

        pred = net(x)

        loss_value = loss(pred, y)
        loss_arr += [loss_value.item()]

        print('train : epoch %04d / %04d | Batch %04d \ %04d | Loss %04f | '
              % (epoch, num_epoch, batch, num_val_for_epoch, np.mean(loss_arr)))

        x = fn_tonumpy(x)
        output = fn_tonumpy(pred)

        writer_val.add_image(
            'input', x, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
        writer_val.add_image(
            'output', output, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch, flag=0)

writer_train.close()
wrtier_val.close()
