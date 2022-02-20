class DiceLoss(nn.Module):

    def __init__(self, tolerance=1e-8):
        super(DiceLoss, self).__init__()
        self.tolerance = tolerance

    def forward(self, pred, label):

        intersection = torch.sum(pred * label) + self.tolerance
        union = torch.sum(pred) + torch.sum(label) + self.tolerance
        dice_loss = 1 - 2 * intersection / union

        return dice_loss

class EnsembleLoss(nn.Module):

    def __init__(self, mask_loss, loss_func):
        super(EnsembleLoss, self).__init__()

        self.mask_loss = mask_loss
        self.loss_func = loss_func

    def forward(self, masks, ensemble_mask, mask):

        ensemble_loss = 0
        num_backbones = len(masks)
        for i in range(num_backbones):
            ensemble_loss  = ensemble_loss + self.mask_loss(masks[i], mask) / num_backbones
        ensemble_loss = 0.5 * ensemble_loss + 0.5 * self.loss_func(ensemble_mask, mask)

        return ensemble_loss
