import paddle
def mean_squared_error(y_true, y_pred):
    return paddle.mean(paddle.square(y_true - y_pred))

def get(identifier):
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    loss_identifier = {
        "mean squared error": mean_squared_error,
        "MSE": mean_squared_error,
        "mse": mean_squared_error,
    }

    if isinstance(identifier, str):
        return loss_identifier[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)
