import os
import time
import torch
import torchsde  
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_data_sde(system, drift, diffusion, n_samples=None, num_simulations=10, v_max=200, 
                      overwrite=False, plot=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = system.domain
    d = len(domain)  

    T = 10.0
    N = 1000  
    t = torch.linspace(0, T, N, device=device)  

    # transfer function 0.1 * ||x||²
    def hh(x):
        # x shape (..., d)
        return 0.1 * torch.norm(x, dim=-1)**2

    # vectorized h：input X shape (N, n_samples, num_simulations, d)
    # return shape (n_samples, num_simulations)，intrgration for h (T/N)*∑_{i=0}^{N-1} hh(X[i, ...])
    def h_vectorized(X):
        hs = hh(X)
        return torch.sum(hs, dim=0) * (T / N)

    class SDEFunc(torchsde.SDEIto):
        def __init__(self, noise_type='general'):
            super().__init__(noise_type=noise_type)
        
        def f(self, t, x):
            # x (batch_size, d)
            return drift(x, t)
        
        def g(self, t, x):
            return diffusion(x, t)

    def simulate_sde_gpu_batch(x_train_tensor, num_simulations):
        # x_train_tensor:  (n_samples, d)
        n_samples = x_train_tensor.shape[0]
        # copy num_simulations, (n_samples, num_simulations, d)
        x0_expanded = x_train_tensor.unsqueeze(1).repeat(1, num_simulations, 1)
        # reshape (n_samples * num_simulations, d)
        x0_flat = x0_expanded.reshape(-1, d)
        
        sde_func = SDEFunc()
        # solve SDE ，X shape (N, n_samples*num_simulations, d)
        X = torchsde.sdeint(sde_func, x0_flat, t, method='euler')
        # reshape (N, n_samples, num_simulations, d)
        X = X.reshape(N, n_samples, num_simulations, d)
        
        # final state of each trajectory, with shape (n_samples, num_simulations, d)
        last_row = X[-1, :, :, :]
        # (n_samples, num_simulations)
        norms = torch.norm(last_row, dim=-1)
        
        # (n_samples, num_simulations)
        integral_h = h_vectorized(X)
        
        # Determine the validity of each trajectory
        mask = (~torch.isnan(norms)) & (norms <= 100) & (~torch.isnan(integral_h)) & (integral_h <= 100)
        # If the conditions are met，v = 1 - exp(-integral_h)，else v = 1
        v = torch.where(mask, 1 - torch.exp(-integral_h), torch.ones_like(integral_h))
        
        # Take the average of num_Simulations for each initial condition, and obtain the shape of the result (n_samples,)
        v_avg = torch.mean(v, dim=-1)
        return v_avg

    # Batch computing training data: Process all initial conditions at once
    def generate_train_data_batch(x_train):
        # x_train (n_samples, d)
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train_tensor = simulate_sde_gpu_batch(x_train_tensor, num_simulations)
        return y_train_tensor

    if not os.path.exists('results'):
        os.makedirs('results')
    t_filename = (f'results/{system.name}_data_{n_samples}_samples_{num_simulations}_simulations'
                  f'_v_max_{v_max}.npy')

    print('_' * 50)
    print("Generating training data from GPU-accelerated simulation:")
    if os.path.exists(t_filename) and not overwrite:
        print("Data exists. Loading training data...")
        t_data = np.load(t_filename)
        x_train, y_train = t_data[:, :-1], t_data[:, -1]
    else:
        print("Generating new training data...")
        start_time = time.time()    
        # Generate initial conditions on the CPU with the shape of (n_samples, d)
        # x_train = np.array([np.random.uniform(dim[0], dim[1], n_samples) 
        #                     for dim in domain]).T
        # print(x_train.shape)
        # y_train_tensor = generate_train_data_batch(x_train)
        # y_train = y_train_tensor.cpu().numpy()
        # print(y_train.shape)
        x_train = np.array([np.random.uniform(dim[0], dim[1], n_samples) 
                    for dim in domain]).T
        # print(x_train.shape)  # (n_samples, d)

        length = 4
        batch_size = x_train.shape[0] // length  # Determine batch size
        y_train_parts = []  # List to store batch results

        # Split x_train into 'length' batches and process each batch
        for i in range(length):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            x_batch = x_train[start_idx:end_idx]  # Extract batch (100, 4)
            y_batch_tensor = generate_train_data_batch(x_batch)  # Process batch
            y_batch = y_batch_tensor.cpu().numpy()  # Convert to numpy
            y_train_parts.append(y_batch)  # Store batch result

        # Combine all batches into a single array
        y_train = np.concatenate(y_train_parts, axis=0)
        # print(y_train.shape)  # (n_samples,)
        print("Saving training data...")
        t_data = np.column_stack((x_train, y_train))
        np.save(t_filename, t_data)
        end_time = time.time()
        print(f"Time taken for generating data: {end_time - start_time} seconds.\n")

    if plot:  
        plt.figure()    
        if d == 1:
            plt.scatter(x_train, np.zeros_like(x_train), c=y_train, cmap='coolwarm', s=1)  
        else:
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='coolwarm', s=1)  
        plt.savefig(f"results/{system.name}_data_{n_samples}_samples_{num_simulations}_simulations_v_max_{v_max}.pdf")
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], y_train, c=y_train, cmap='viridis')
        ax.set_xlabel('x0[0]')
        ax.set_ylabel('x0[1]')
        ax.set_zlabel('v(x0)')
        ax.set_title(f'3D Plot of v(x0)')
        plt.savefig(f"results/{system.name}_data_{n_samples}_samples_{num_simulations}_simulations_3D_v_max_{v_max}.pdf")
        plt.close()

    return np.column_stack((x_train, y_train))
