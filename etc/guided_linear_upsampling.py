import numpy as np
import matplotlib.pyplot as plt

def initialize_I_L(I):
    # Implement regular grid downsampling
    return I[::2, ::2]

def find_most_similar_pixel(I, I_L, p):
    # Find the pixel in Ω^L_p with the most similar color to I_p
    min_diff = float('inf')
    a = None
    for i in range(max(0, p[0]-1), min(I_L.shape[0], p[0]+2)):
        for j in range(max(0, p[1]-1), min(I_L.shape[1], p[1]+2)):
            diff = np.linalg.norm(I[p] - I_L[i, j])
            if diff < min_diff:
                min_diff = diff
                a = (i, j)
    return a

def optimize_b_omega(I, I_L, T_L, a):   #####  수정 필요 
    # Optimize b and ω_ab
    b = np.mean(I - I_L)
    omega_ab = 1.0  # Placeholder value
    return b, omega_ab

def compute_T_hat(T_L, a, b, omega_ab):
    # Compute T^ as the given equation
    T_hat = omega_ab * T_L[a] + (1 - omega_ab) * T_L[b]
    return T_hat

def efficient_guided_linear_upsampling(I, I_L, T_L):
    T_hat = np.zeros_like(I)
    Theta = {}
    for p in np.ndindex(I.shape):
        a = find_most_similar_pixel(I, I_L, p)
        b, omega_ab = optimize_b_omega(I, I_L, T_L, a)
        T_hat[p] = compute_T_hat(T_L, a, b, omega_ab)
        Theta[p] = (a, b, omega_ab)
    return T_hat, Theta

def initialize_Theta(I, I_L):
    # Initialize Θ from I, I_L with Algorithm 1
    T_L = np.zeros_like(I_L)  # Placeholder for the low-res target image
    _, Theta = efficient_guided_linear_upsampling(I, I_L, T_L)
    return Theta

def compute_initial_error_map(I, I_L, Theta):
    # Compute the initial error map E
    E = np.abs(I - I_L)
    return E

def update_I_L_q(I_L, I, q, C_i, E):
    # Update I_L at q with I at p where p maximizes E_p
    E_q_neighbors = E[q[0]-1:q[0]+2, q[1]-1:q[1]+2]
    max_error_idx = np.unravel_index(np.argmax(E_q_neighbors), E_q_neighbors.shape)
    p = (q[0] + max_error_idx[0] - 1, q[1] + max_error_idx[1] - 1)
    I_L[q] = I[p]
    return I_L, p

def update_Theta_p(Theta, I, I_L, T_L, p):
    # Update Theta_p as in Algorithm 1
    a = find_most_similar_pixel(I, I_L, p)
    b, omega_ab = optimize_b_omega(I, I_L, T_L, a)
    Theta[p] = (a, b, omega_ab)
    return Theta[p]

def joint_optimization(I, tau, N):
    I_L = initialize_I_L(I)
    Theta = initialize_Theta(I, I_L)
    E = compute_initial_error_map(I, I_L, Theta)

    for n in range(N):
        E_set = np.argwhere(E > tau)
        if len(E_set) == 0:
            break

        # Cluster E as connected components (placeholder)
        clusters = [E_set]

        for C_i in clusters:
            # Backup Θ, I^L, E for scroll back
            Theta_backup = Theta.copy()
            I_L_backup = I_L.copy()
            E_backup = E.copy()

            e0 = np.sum(E[C_i[:, 0], C_i[:, 1]])
            Q = set()
            for p in C_i:
                neighbors = [(p[0] + i, p[1] + j) for i in range(-1, 2) for j in range(-1, 2)]
                Q.update(neighbors)
            Q = Q.intersection(set([tuple(q) for q in np.argwhere(I_L)]))

            for q in Q:
                I_L, p = update_I_L_q(I_L, I, q, C_i, E)
            
            for p in C_i:
                Theta[p] = update_Theta_p(Theta, I, I_L, T_L, p)
                E[p] = np.abs(I[p] - I_L[p])  # Update E_p with updated Θ_p
            
            e1 = np.sum(E[C_i[:, 0], C_i[:, 1]])
            if e1 > e0:
                # Scroll back updated regions of Θ, I^L, and E
                Theta = Theta_backup
                I_L = I_L_backup
                E = E_backup

    return I_L, Theta

# Example usage
high_res_image = np.random.rand(512, 512, 3)  # Placeholder for a high-resolution image
tau = 0.1  # Error threshold
N = 10  # Maximum number of iterations

# Run the joint optimization algorithm
I_L, Theta = joint_optimization(high_res_image, tau, N)

# Output the results
print("Optimized low-res image I_L:")
print(I_L)

print("\nUpsampling parameters Theta:")
print(Theta)

# Visualize the results

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("High-res Image")
plt.imshow(high_res_image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Low-res Image I_L")
plt.imshow(I_L)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Theta (Upsampling Parameters)")
plt.imshow(np.mean(list(Theta.values()), axis=0))  # Visualize mean of Theta values
plt.axis('off')

plt.show()
