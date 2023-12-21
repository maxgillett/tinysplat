from tinygrad import Tensor

def logit(x):
    return Tensor.log(x/(1-x))

def num_sh_bases(degree: int):
    if degree == 0:
        return 1
    if degree == 1:
        return 4
    if degree == 2:
        return 9
    if degree == 3:
        return 16
    return 25

