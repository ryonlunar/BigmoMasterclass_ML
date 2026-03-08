import math


def linear():
    pass

def relu():
    pass
    
# softmax
def softmax(inputs: list[float]) -> list[float]:
    exp_inputs = [math.exp(xi) for xi in inputs]
    sum_exp_inputs = sum(exp_inputs)
    return [ei / sum_exp_inputs for ei in exp_inputs]