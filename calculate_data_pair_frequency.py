def calculate_probabilities(N_d, N_w, N_a, b_d, b_w, T, T_a, n_T, n_T_a):
    # Activated phase probabilities
    P_d1 = (b_d * (b_d - 1)) / (N_d * (N_d - 1)) * (T * n_T + T_a * n_T_a)
    P_w1 = (b_w * (b_w - 1)) / (N_w * (N_w - 1)) * T * n_T
    P_a1 = (b_w * (b_w - 1)) / (N_a * (N_a - 1)) * T_a * n_T_a

    # Dormant phase sizes
    N_d_prime = N_d - b_d * (T * n_T + T_a * n_T_a)
    N_w_prime = N_w - b_w * T * n_T
    N_a_prime = N_a - b_w * T_a * n_T_a
    N_prime = N_d_prime + N_w_prime + N_a_prime

    # Dormant phase probabilities
    P_d2 = (N_d_prime * (N_d_prime - 1)) / (N_prime ** 2 * (N_prime - 1))
    P_w2 = (N_w_prime * (N_w_prime - 1)) / (N_prime ** 2 * (N_prime - 1))
    P_a2 = (N_a_prime * (N_a_prime - 1)) / (N_prime ** 2 * (N_prime - 1))

    # Final probabilities
    P_d = P_d1 + P_d2
    P_w = P_w1 + P_w2
    P_a = P_a1 + P_a2

    return P_d, P_w, P_a

def calculate_cross_probabilities(N_d, N_w, N_a, b_d, b_w, T, T_a, n_T, n_T_a):
    # Activated Phase
    P_dw1 = (b_d / N_d) * (b_w / N_w) * (T * n_T)
    P_da1 = (b_d / N_d) * (b_w / N_a) * (T_a * n_T_a)
    P_wa1 = 0

    # Dormant phase
    N_d_prime = N_d - b_d * (T * n_T + T_a * n_T_a)
    N_w_prime = N_w - b_w * T * n_T
    N_a_prime = N_a - b_w * T_a * n_T_a
    N_prime = N_d_prime + N_w_prime + N_a_prime
    b = b_d + b_w

    P_dw2 = b / N_prime

    P_wa2 = P_dw2  # Using symmetry
    P_da2 = P_dw2  # Using symmetry

    # Combined Probabilities
    P_dw = P_dw1 + P_dw2
    P_wa = P_wa1 + P_wa2
    P_da = P_da1 + P_da2

    print("For one epoch, the cross-probabilities are:")
    print(f"Clean-watermark pair probability: {P_dw}")
    print(f"Watermark-anti pair probability: {P_wa}")
    print(f"clean—anti pair probability: {P_da}")


def calculate_single_probabilties(N_d, N_w, N_a, b_d, b_w, T, T_a, n_T, n_T_a):
    P_d = (b_d)/N_d
    P_w = (b_w)/N_w
    P_a = 0

    P_d = (b_d) / N_d
    P_a = (b_w) / N_a
    P_w = 0

    N_d_prime = N_d - b_d * (T * n_T + T_a * n_T_a)
    N_w_prime = N_w - b_w * T * n_T
    N_a_prime = N_a - b_w * T_a * n_T_a
    N_prime = N_d_prime + N_w_prime + N_a_prime
    b = b_d + b_w
    S_d = b*(N_d_prime/N_prime)
    S_w = b*(N_w_prime/N_prime)
    S_a = b*(N_a_prime/N_prime)
    P_d = (S_d) / N_d
    P_a = (S_a) / N_a
    P_w = (S_w) / N_a

    print(f"Clean sample probability: {P_d}")
    print(f"Watermark sample probability: {P_w}")
    print(f"Anti sample probability: {P_a}")

# Example usage
N_d, N_w = 30000, 1800
N_a = 39209-N_d-N_w  # Example dataset sizes
b_d, b_w = 13, 3  # Batch sizes
T, T_a = 30, 300  # Durations
n_T, n_T_a = 4, 4  # Number of repetitions

P_d, P_w, P_a = calculate_probabilities(N_d, N_w, N_a, b_d, b_w, T, T_a, n_T, n_T_a)
print(f"Clean pair probability: {P_d}")
print(f"Watermark pair probability: {P_w}")
print(f"Anti-watermark pair probability: {P_a}")
