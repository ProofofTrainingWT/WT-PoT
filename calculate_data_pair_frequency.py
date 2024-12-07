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
