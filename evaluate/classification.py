# classification.py


class Classification():
    def __init__(self, topk=(1,)):
        self.topk = topk

    def forward(self, output, target):
        """Computes the precision@k for the specified values of k"""
        maxk = max(self.topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
