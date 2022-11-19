import numpy as np


def main():
    network_size = 50
    num_patterns = 3

    print("Creating the weights matrix...")
    patterns = 2 * np.random.randint(0, 2, (num_patterns, network_size)) - 1
    weights_matrix = np.dot(np.transpose(patterns), patterns) / len(patterns)
    np.fill_diagonal(weights_matrix, np.zeros_like(np.diag(weights_matrix)))

    diagonal_is_0 = (np.diag(weights_matrix) == 0).all()
    is_symmetric = (np.transpose(weights_matrix) == weights_matrix).all()
    between_minus_1_and_1 = (weights_matrix <= 1).all() and (weights_matrix >= -1).all()

    print("The elements on the diagonal are 0:", diagonal_is_0)
    print("The matrix is symmetric:", is_symmetric)
    print("All the elements are between -1 and 1:", between_minus_1_and_1)

    base_pattern = 0
    num_perturb = 10
    max_iter = 20

    print(f"Retrieving the memorized pattern number {base_pattern}...")
    state = patterns[base_pattern, :].copy()
    idx = np.random.choice(list(range(network_size)), size=num_perturb, replace=False)
    state[idx] = - state[idx]
    state_list = [state]

    state_old = np.zeros_like(state)
    i = 0
    while (state_old != state).any() and i < max_iter:
        state_old = state
        state = np.dot(weights_matrix, state)
        state = np.where(state >= 0, 1, -1)
        state_list.append(state)
        i += 1
    if i == max_iter:
        print(f"The process did not converge in {i} iterations")
    else:
        print(f"The process converged in {i} iterations")

    pattern_match = np.where([(p == state).all() for p in patterns])[0]
    if pattern_match.size > 0:
        print(f"The network converged to the pattern nr. {pattern_match[0]}")
    else:
        print(F"The network did not converge to any of the stored patterns")

if __name__ == "__main__":
    main()