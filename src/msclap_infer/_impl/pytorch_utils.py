"""Only the time interpolation operation used by the HTSAT inference path."""


def interpolate(x, ratio):
    batch_size, time_steps, classes_num = x.shape
    return x[:, :, None, :].repeat(1, 1, ratio, 1).reshape(
        batch_size, time_steps * ratio, classes_num
    )
