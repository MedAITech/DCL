import torch.nn as nn
import network
import torch
class CrowdCounter(nn.Module):
    def __init__(self,model,pool):
        super(CrowdCounter, self).__init__()
        if model=='base':
            from base import base
            self.DME = base(pool)
        if model=='wide':
            from base import base
            self.DME = base(pool)
        if model=='deep':
            from base import base
            self.DME = base(pool)

        self.loss_fn = nn.MSELoss()

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, im_data, gt_data=None):
        density_map = self.DME(im_data)

        return density_map

   

if __name__ == '__main__':
    x = torch.rand(7, 3, 304, 304)
    model = CrowdCounter(model='base', pool='vpool')
    output = model(x)
    print(output.shape)