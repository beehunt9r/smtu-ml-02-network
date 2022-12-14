import numpy as np


def generate_weights(count: int) -> np.array:
    return np.random.uniform(-1.0, 1.0, count)


def compute_output(x: np.array, w: np.array) -> float:
    z = np.dot(x, w)
    return np.sign(z)


def compute_operation(x: np.array, w: np.array) -> float:
    return compute_output(
        np.concatenate(([1.0], x)),
        w
    )


def compute_xor_operation(x: np.array, not_w: np.array, and_w: np.array, or_w: np.array) -> float:
    # XOR(x1, x2) = AND(
    #   NOT(AND(x1, x2)),
    #   OR(x1, x2)
    # )
    and_operation = compute_operation(x, and_w)
    not_operation = compute_operation(np.array([and_operation]), not_w)

    or_operation = compute_operation(x, or_w)

    return compute_operation(
        np.array([not_operation, or_operation]),
        and_w
    )


def compute_nor_operation(x: np.array, not_w: np.array, or_w: np.array) -> float:
    # NOR(x1, x2) = NOT(OR(x1, x2))
    or_operation = compute_operation(x, or_w)

    return compute_operation(
        np.array([or_operation]),
        not_w
    )


def add_line_to_subplot(subplot, weights, label=None):
    x = [-2.0, 2.0]
    if len(weights) < 3 or abs(weights[2]) < 1e-5:
        y = [-weights[1] / 1e-5 * x[0] + (-weights[0] / 1e-5),
             -weights[1] / 1e-5 * x[1] + (-weights[0] / 1e-5)]
    else:
        y = [-weights[1] / weights[2] * x[0] + (-weights[0] / weights[2]),
             -weights[1] / weights[2] * x[1] + (-weights[0] / weights[2])]

    subplot.plot(x, y, label=label)
