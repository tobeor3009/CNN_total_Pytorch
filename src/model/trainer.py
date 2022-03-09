from torch.utils.tensorboard import SummaryWriter

start_epoch = 0
for epoch in range(start_epoch + 1, num_epoch + 1):

    net.train()
    loss_arr = []

    for batch, data in enumerate(train_dl, 1):
        x = data['img'].cuda()
        y = data['mask'].cuda()

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
        for batch, data in enumerate(val_dl, 1):
            x = data['img'].to(device)
            y = data['mask'].to(device)

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
