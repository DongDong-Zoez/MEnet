class EnsembleHead(nn.Module):

      def __init__(self, in_dims, out_dims):
          super(EnsembleHead, self).__init__()
          self.conv3x3 = nn.Conv2d(in_dims, out_dims, 3, 1, 1)
          self.convmid = nn.Conv2d(out_dims, out_dims, 3, 1, 1)
          self.conv1x1 = nn.Conv2d(out_dims, 3,  1, 1, 0)
          self.bn = nn.BatchNorm2d(out_dims)
          self.relu = nn.ReLU()
          self.quantile = torch.tensor([0.1])

      def forward(self, x):

          x = self.conv3x3(x)
          x = self.bn(x)
          x = self.relu(x)
          x = self.convmid(x)
          x = self.bn(x)
          x = self.relu(x)
          x = self.conv1x1(x)

          return x

class MEnet(nn.Module):

    def __init__(self, modules):
        super(MEnet, self).__init__()
        self.module_list = nn.ModuleList(modules)
        self.num_module = len(self.module_list)
        self.in_dim = self.get_in_dim(self.module_list[0])

        self.ensemble_layer = EnsembleHead(self.in_dim, 64)

    def get_in_dim(self, module):

      x = torch.randn(1, 3, 128, 128)
      X = module(x)

      return x.shape[1]

    def forward(self, x):

        masks = []
        for module in self.module_list:
            masks.append(module(x))

        ensemble_mask = torch.cat(masks, dim=1)
        ensemble_mask = sum(masks) / self.num_module
        ensemble_mask = self.ensemble_layer(ensemble_mask)

        return masks, ensemble_mask

    def toDevice(self, device):
        for i in range(self.num_module):
            self.module_list[i].to(device=device)
