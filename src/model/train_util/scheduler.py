from functools import partial
from torch.optim.lr_scheduler import LambdaLR

def one_cycle_lr(step, step_size,
                 first_epoch=10, second_epoch=90, total_epoch=200,
                 max_min_lr_ratio=5, decay_lr=0.25):
    first_step = step_size * first_epoch
    second_step = step_size * second_epoch
    total_step = step_size * total_epoch
    if step < first_step:
        # 첫 10 에포크 동안 min => max
        return 1 + (max_min_lr_ratio - 1) * (step / first_step)
    elif step < first_step + second_step:
        # 그 다음 90 에포크 동안 max => min
        return max_min_lr_ratio - (max_min_lr_ratio - 1) * ((step - first_step) / second_step)
    else:
        step = min(total_step, step)
        # 그 후 50 에포크마다 0.5배 감소
        factor = 1 - (1 - decay_lr) * ((step - first_step - second_step) / (total_step - first_step - second_step))
        return factor
    

class OneCycleLR(LambdaLR):
    def __init__(self, optimizer, step_size,
                 first_epoch=10, second_epoch=90, total_epoch=200,
                 max_min_lr_ratio=5, decay_lr=0.25):

        assert max_min_lr_ratio > 1, "max_min_lr_ratio should be greater than 1. \
            It is recommended to ensure the maximum learning rate exceeds the minimum learning rate \
                for better training efficiency."
        self.step_size = step_size
        lr_lambda = partial(one_cycle_lr, step_size=step_size,
                            first_epoch=first_epoch, second_epoch=second_epoch, total_epoch=total_epoch,
                            max_min_lr_ratio=max_min_lr_ratio, decay_lr=decay_lr)
        super().__init__(optimizer, lr_lambda=lr_lambda)

    def set_epoch(self, epoch):
        self._step_count = self.step_size * epoch
        values = self.get_lr()
        
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

