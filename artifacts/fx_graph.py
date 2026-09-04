


def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    sin = l_x_.sin();  l_x_ = None
    mul = sin * 2;  sin = None
    relu = torch.relu(mul);  mul = None
    return (relu,)
    