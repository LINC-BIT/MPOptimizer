import torch


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification (from US-Net) """
    def forward(self, output, target):
        target = torch.nn.functional.softmax(target, dim=1)
        
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob).mean()
        return cross_entropy_loss

class CrossEntropyLossSoft2(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification (from US-Net) """
    def forward(self, output, target, loss_mask):
        target = torch.nn.functional.softmax(target, dim=-1)
        
        output_log_prob = torch.nn.functional.log_softmax(output, dim=-1)
        target = target.unsqueeze(-2)
        output_log_prob = output_log_prob.unsqueeze(-1)
        cross_entropy_loss = -torch.matmul(target, output_log_prob).squeeze(-1).squeeze(-1)
        num_loss = loss_mask.sum()
        loss = torch.sum(cross_entropy_loss.view(-1) * loss_mask.view(-1)) / num_loss
        return loss