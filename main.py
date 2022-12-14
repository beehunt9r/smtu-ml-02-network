import math

from functions import *
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt

# Define learning rate for train.
np.random.seed(31072003)
learning_rate = 0.35

# Create dataset with x (input) and y (output).
data = {
    'not': {
        'x': (
            np.array([1.0, 1.0]),
            np.array([1.0, -1.0])
        ),
        'y': (-1.0, 1.0)
    },
    'and': {
        'x': (
            np.array([1.0, 1.0, -1.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, -1.0, 1.0]),
            np.array([1.0, -1.0, -1.0])
        ),
        'y': (-1.0, 1.0, -1.0, -1.0)
    },
    'or': {
        'x': (
            np.array([1.0, 1.0, -1.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, -1.0, 1.0]),
            np.array([1.0, -1.0, -1.0])
        ),
        'y': (1.0, 1.0, 1.0, -1.0)
    }
}

# Create empty array for weights.
weights = {}

# Create empty array for plots.
figure = plt.figure()
figure.subplots_adjust(hspace=0.4, wspace=0.4)

plots = {}

# Train each operation.
for operation, dataset in data.items():
    print(f'Training "{operation}" operation...')

    # If input and output lengths are different, raise error, otherwise generate indexes list.
    x_length = len(dataset['x'])
    y_length = len(dataset['y'])

    if x_length != y_length:
        raise Exception(f'Operation "{operation}" has different input (x = {x_length}) and '
                        f'output (y = {y_length}) lengths!')
    elif x_length == 0:
        raise Exception(f'Operation "{operation}" has empty dataset!')

    dataset_indexes = np.array(range(x_length))

    # Fill operation with random weights.
    values_count = len(dataset['x'][0])
    weights[operation] = generate_weights(values_count)

    # Create subplot for operator.
    rows = columns = math.ceil(math.sqrt(len(data)))
    plots[operation] = subplot = figure.add_subplot(rows, columns, list(data.keys()).index(operation) + 1)

    # Set default subplot options.
    subplot.set_title(f'"{operation}" perceptron')
    subplot.set_xlim((-1.2, 1.2))
    subplot.set_ylim((-1.2, 1.2))

    # Show dataset values on subplot.
    for index in dataset_indexes:
        x = dataset['x'][index]
        y = dataset['y'][index]

        subplot.plot(
            x[1],
            0.0 if len(x) < 3 else x[2],
            'r+' if y == 1.0 else 'b_'
        )

    # Train perceptron on current operation dataset.
    iterations_count = 0
    trained = False
    w = weights[operation]
    while not trained:
        trained = True

        # Shuffle indexes to iterate random.
        np.random.shuffle(dataset_indexes)
        for index in dataset_indexes:
            x = dataset['x'][index]
            y = dataset['y'][index]

            # Compute output by dataset data.
            y_computed = compute_output(x, w)

            if y_computed != y:
                # If output is wrong, update weights.
                for w_index in range(len(w)):
                    w[w_index] += y * learning_rate * x[w_index]
                trained = False

                # Increment iterations count.
                iterations_count += 1

                # Show calculated weights.
                print(f'{iterations_count}. {w}')

                # Add trained line to plot.
                add_line_to_subplot(subplot, w)

    # Add correct line to plot.
    add_line_to_subplot(subplot, w, '+')
    subplot.legend()

    print(f'Trained successfully in {iterations_count} iteration(s), {w}\n')

print('All perceptron\'s trained successfully!')

# Generate final result table.
table = pt()
table.title = 'Variant №1'
table.field_names = [f'x{index}' for index in range(1, 5)] + ['NOT', 'OR', 'XOR', 'NOR']

# F(x1, x2, x3, x4) = NOR(
#  x4,
#  XOR(
#   OR(x3, x4),
#   NOT(x1)
#  )
# )
for x1 in [-1.0, 1.0]:
    for x2 in [-1.0, 1.0]:
        for x3 in [-1.0, 1.0]:
            for x4 in [-1.0, 1.0]:
                not_operation = compute_operation(np.array([x1]), weights['not'])
                or_operation = compute_operation(
                    np.array([x3, x4]),
                    weights['or']
                )

                xor_operation = compute_xor_operation(
                    np.array([not_operation, or_operation]),
                    weights['not'],
                    weights['and'],
                    weights['or']
                )

                nor_operation = compute_nor_operation(
                    np.array([x4, xor_operation]),
                    weights['not'],
                    weights['or']
                )

                table.add_row([
                    x1, x2, x3, x4,
                    not_operation, or_operation, xor_operation,
                    nor_operation
                ])

print(table)

plt.show()
