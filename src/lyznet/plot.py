from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import torch
import numpy as np
import os
from scipy.integrate import solve_ivp
import random
import pandas as pd
import lyznet


rc('font', **{'family': 'Linux Libertine O'})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'stix'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_accumulated_cost(system, model_path, elm_model, u_func, Q, R, T=10, 
                          step_size=0.02, save_csv=True, 
                          closed_loop_f_numpy=None):
    domain = system.domain

    if system.system_type == "DynamicalSystem":
        f_np = lambda t, x: system.f_numpy_vectorized(x) 
    else: 
        # For non-autonomous systems, ensure closed_loop_f_numpy is provided
        if closed_loop_f_numpy is None:
            raise ValueError("The system is not autonomous, "
                             "but a closed-loop vector field is not given.")
        f_np = lambda t, x: closed_loop_f_numpy(x)

    initial_conditions = [d[0]/3 for d in domain]

    fig, ax = plt.subplots()

    # Solve the differential equation from the given initial conditions
    sol = solve_ivp(f_np, [0, T], initial_conditions, method='RK45', 
                    t_eval=np.linspace(0, T, int(T/step_size)))

    # Calculate cost at each time step
    cost_values = []
    for x in sol.y.T:
        x_np = np.array(x)
        # print("x: ", x_np.shape)
        if elm_model is None:
            x_tensor = torch.tensor(x_np, dtype=torch.float32).view(1, -1)
            u_np = u_func(x_tensor).squeeze(0).detach().numpy()
        else: 
            u_np = u_func(x_np).T
        # print("Q: ", Q.shape)
        # print("x: ", x_np.shape)
        # print("R: ", R.shape)
        # print("u: ", u_np.shape)        
        cost = x_np.T @ Q @ x_np + u_np.T @ R @ u_np
        cost_values.append(cost)

    accumulated_cost = np.cumsum(np.array(cost_values) * step_size)

    ax.plot(sol.t, accumulated_cost, label='Accumulated Cost')

    ax.set_xlabel('Time')
    ax.set_ylabel('Accumulated Cost')
    ax.set_title('Accumulated Cost Over Time')

    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'{model_path}_cost.pdf', format='pdf', dpi=300)
    plt.close(fig)

    if save_csv:
        cost_data = pd.DataFrame({'Time': sol.t, 'Cost': accumulated_cost})
        csv_file_path = f'{model_path}_cost.csv'
        cost_data.to_csv(csv_file_path, index=False)


def simulate_trajectories(system, model_path, n=50, T=10, 
                          closed_loop_f_numpy=None):
    print(f"Simulating {n} trajectories from random intitial conditions...")
    fig, ax = plt.subplots() 
    domain = system.domain  
    if system.system_type == "DynamicalSystem":
        f = lambda t, x: system.f_numpy_vectorized(x) 
    elif system.system_type == "StochasticDynamicalSystem":
        # Handle stochastic dynamical systems
        f = lambda t, x: system.stochastic_derivative_numpy_vectorized(x)
    else:
        # For non-autonomous systems, ensure closed_loop_f_numpy is provided
        if closed_loop_f_numpy is None:
            # raise ValueError("The system is not autonomous, "
            #                  "but a closed-loop vector field is not given.")
            closed_loop_f_numpy = system.closed_loop_f_numpy
        f = lambda t, x: closed_loop_f_numpy(x)

    lyznet.tik()
    for _ in range(n):
        # Random initial conditions within the full domain
        # initial_conditions = [random.uniform(*d) for d in domain]

        # Try initial conditions within a smaller domain
        half_domain = [[d[0] / 3, d[1] / 3] for d in domain]
        initial_conditions = [random.uniform(*d) for d in half_domain]

        # Solve differential equation using RK45
        sol = solve_ivp(f, [0, T], initial_conditions, method='RK45', 
                        t_eval=np.linspace(0, T, 500))
        # Plot each dimension of the solution against time
        for i in range(sol.y.shape[0]):
            ax.plot(sol.t, sol.y[i], linewidth=1, 
                    label=f'Dimension {i+1}' if _ == 0 else "")

    ax.set_xlabel('Time')
    ax.set_ylabel('States')
    ax.set_title('Trajectories of All States Over Time')

    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'{model_path}_{n}_trajectories_[0,{T}].pdf', 
                format='pdf', dpi=300)
    plt.close(fig)
    lyznet.tok()


def evaluate_vector_field(x1, x2, system, closed_loop_f_numpy):
    d = len(system.symbolic_vars)
    zero_dims = d - 2
    x_dims = [np.zeros_like(x1) for _ in range(zero_dims)]
    input_array = np.vstack(
        [x1.ravel(), x2.ravel()] + [x_dim.ravel() for x_dim in x_dims]
    ).T
    # print("input_array: ", input_array.shape)
    # f_values = [f_np(*input_array.T) for f_np in system.f_numpy]
    # f_values = system.f_numpy(*input_array.T)

    if system.system_type == "DynamicalSystem":
        f_values = system.f_numpy_vectorized(input_array).T 
        # print(f_values.shape)
    elif system.system_type == "StochasticDynamicalSystem":
        # Handle stochastic systems
        f_values = system.f_numpy_vectorized(input_array).T 
    else:
        # For non-autonomous systems, ensure closed_loop_f_numpy is provided
        if closed_loop_f_numpy is None:
            # raise ValueError("The system is not autonomous, "
            #                  "but a closed-loop vector field is not given.")
            closed_loop_f_numpy = system.closed_loop_f_numpy
        f_values = closed_loop_f_numpy(input_array).T
    f_values = np.array(f_values)[:2]  # Only taking first two dimensions
    # print("f_values: ", f_values.shape)
    return f_values[0].reshape(x1.shape), f_values[1].reshape(x2.shape)


def plot_phase_portrait(Xd, Yd, system, closed_loop_f_numpy=None):
    DX, DY = evaluate_vector_field(Xd, Yd, system, closed_loop_f_numpy)
    # print("DX: ", DX)
    # print("DY: ", DY)
    plt.streamplot(Xd, Yd, DX, DY, color='gray', linewidth=0.5, density=0.8, 
                   arrowstyle='-|>', arrowsize=1)


def evaluate_dynamics(f, x):
    x_split = torch.split(x, 1, dim=1)
    result = []
    for fi in f:
        args = [x_s.squeeze() for x_s in x_split]
        result.append(fi(*args))
    return result


def plot_lie_derivative(system, net, x_tensor, x1, x2, ax):
    # Calculate V and its gradient
    x_tensor.requires_grad = True
    V = net(x_tensor).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x_tensor, create_graph=True)[0]

    if system.system_type == "DynamicalSystem":
        # Calculate dynamics and V_dot
        f_values = evaluate_dynamics(system.f_torch, x_tensor)
        f_tensor = torch.stack(f_values, dim=1)
        V_dot = (V_grad * f_tensor).sum(dim=1)

        # Reshape V_dot for plotting
        V_dot_reshaped = V_dot.detach().cpu().numpy().reshape(x1.shape)

        # Overlay the lie derivative plot on the same axes
        ax.plot_surface(x1, x2, V_dot_reshaped, color='red', alpha=0.5, 
                        label="Lie Derivative")
    elif system.system_type == "StochasticDynamicalSystem":
        # Calculate dynamics and V_dot
        f_values = evaluate_dynamics(system.f_torch,  x_tensor)
        f_tensor = torch.stack(f_values, dim=1)
        g_values = evaluate_dynamics(system.g_torch,  x_tensor)
        if torch.is_tensor(g_values[0][0]):
            first_tensor = g_values[0][0].clone().detach().requires_grad_(True)
        else:
            first_tensor = torch.tensor(g_values[0][0], dtype=torch.float32, requires_grad=True)

        zero_tensor = torch.zeros_like(first_tensor) * first_tensor
        for i in range(len(g_values)):
            for j in range(len(g_values[i])):
                if isinstance(g_values[i][j], int) and g_values[i][j] == 0:
                    g_values[i][j] = zero_tensor
                elif torch.is_tensor(g_values[i][j]) and torch.all(g_values[i][j] == 0):
                    g_values[i][j] = zero_tensor
        # print("g_values:", g_values)
        g_tensor = torch.stack([
        torch.stack([g_values[0][0], g_values[0][1]], dim=1),  
        torch.stack([g_values[1][0], g_values[1][1]], dim=1)   
        ], dim=2)  

        V_xx = torch.zeros( x_tensor.shape[0], 2, 2)
        for i in range(2):
            V_xx[:, i, :] = torch.autograd.grad(V_grad[:, i],  x_tensor, grad_outputs=torch.ones_like(V_grad[:, i]), create_graph=True)[0]
        # print("g_tensor shape:", g_tensor.shape)  #  (batch_size, j, i)
        # print("V_xx shape:", V_xx)  #  (batch_size, j, k)
        V_xx = V_xx.to(g_tensor.device)
        g_transposed = torch.transpose(g_tensor, 1, 2)
        # print("g_transposed shape:", g_transposed.shape) 
        trace_term =  torch.einsum('bji,bjk,bki->b', g_transposed, V_xx, g_tensor)

        V_dot = (V_grad * f_tensor).sum(dim=1) +0.5*trace_term

        # Reshape V_dot for plotting
        V_dot_reshaped = V_dot.detach().cpu().numpy().reshape(x1.shape)

        # Overlay the lie derivative plot on the same axes
        ax.plot_surface(x1, x2, V_dot_reshaped, color='red', alpha=0.5, 
                        label="Lie Derivative")


def plot_obstacles(ax, obstacles):
    for obstacle in obstacles:
        if obstacle['type'] == 'rectangle':
            center = obstacle['center']
            width = obstacle['width']
            height = obstacle['height']
            lower_left_corner = (center[0] - width / 2, center[1] - height / 2)
            rect = Rectangle(lower_left_corner, width, height, color='grey', 
                             alpha=0.7)
            ax.add_patch(rect)
        elif obstacle['type'] == 'circle':
            center = obstacle['center']
            radius = obstacle['radius']
            circle = Circle(center, radius, color='grey', alpha=0.7)
            ax.add_patch(circle)


def plot_V(system, net=None, model_path=None, V_list=None, c_lists=None, 
           c1_V=None, c2_V=None, c1_P=None, c2_P=None, 
           phase_portrait=None, elm_model=None, lie_derivative=None, 
           plot_trajectories=None, n_trajectories=50, plot_cost=None,
           u_func=None, Q=None, R=None, closed_loop_f_numpy=None,
           obstacles=None):
    domain = system.domain
    d = len(system.symbolic_vars)

    print("Plotting learned Lyapunov function and level sets...")
    lyznet.tik()
    if d == 1:
        x1 = np.linspace(*domain[0], 400)
        x1 = x1.reshape(-1, 1)  # Reshape to 2D array for consistency
        input_array = x1
        P = system.P

        def quad_V(x):
            return x.T @ P @ x

        quad_V_test = np.array(
            [quad_V(x) for x in input_array]
            ).reshape(x1.shape)

        if elm_model is not None: 
            weights, bias, beta = elm_model

            def elm_V(x):
                H = np.matmul(x, weights.T) + bias.T
                return np.tanh(H) @ beta
            V_test = np.array(
                [elm_V(x) for x in input_array]
                ).reshape(x1.shape)
        elif net is not None: 
            x_test = torch.tensor(input_array, dtype=torch.float32).to(device)
            V_net = net(x_test)
            V_test = V_net.detach().cpu().numpy().reshape(x1.shape)
        else: 
            V_test = quad_V_test

        fig = plt.figure(figsize=(12, 6))  # Set figure size
        ax1 = fig.add_subplot(121)
        ax1.plot(x1, V_test, label="Learned Lyapunov Function")

        # Plotting level sets for 1D

        if c1_V is not None or c2_V is not None: 
            level_values = [c1_V, c2_V]  # Add other level values if needed
            for level in level_values:
                if level is not None:
                    level_points = x1[(V_test < level) 
                                      & (np.abs(V_test - level) < 1e-2)]
                    ax1.plot(level_points, [level] * len(level_points), 'ro')  

            for x_val in level_points:
                ax1.axvline(x=x_val, color='k', linestyle='--', alpha=0.7)

        ax1.set_xlabel(r"$x_1$", fontsize=24)
        ax1.set_ylabel("V(x)", fontsize=24)
        ax1.set_title("Learned Lyapunov Function")
        ax1.legend()

    else: 
        # Generate samples for contour plot
        x1 = np.linspace(*domain[0], 200)
        x2 = np.linspace(*domain[1], 200)
        x1, x2 = np.meshgrid(x1, x2)
        # Plots are projected to (x1,x2) plane. Set other dimensions to zero.
        zero_dims = d - 2
        x_dims = [np.zeros_like(x1) for _ in range(zero_dims)]
        # Stack x1, x2, and zero dimensions to form the input tensor
        input_array = np.vstack(
            [x1.ravel(), x2.ravel()] + [x_dim.ravel() for x_dim in x_dims]
        ).T

        if system.P is not None:
            P = system.P

            def quad_V(x):
                return x.T @ P @ x

            quad_V_test = np.array([quad_V(x) for x in input_array]).reshape(x1.shape)

        if elm_model is not None: 
            weights, bias, beta = elm_model

            def elm_V(x):
                H = np.matmul(x, weights.T) + bias.T
                return np.tanh(H) @ beta
            V_test = np.array([elm_V(x) for x in input_array]).reshape(x1.shape)
        elif net is not None: 
            x_test = torch.tensor(input_array, dtype=torch.float32).to(device)
            V_net = net(x_test)
            V_test = V_net.detach().cpu().numpy().reshape(x1.shape)
        else: 
            V_test = quad_V_test

        fig = plt.figure(figsize=(12, 6))  # Set figure size

        # Subplot 1: 3D surface plot of the learned function
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_surface(x1, x2, V_test)  # Plot the learned function
        ax1.set_xlabel(r"$x_1$", fontsize=24)
        ax1.set_ylabel(r"$x_2$", fontsize=24)
        # ax1.set_zlabel("V(x)")
        ax1.set_title("Learned Lyapunov Function")

        if net is not None and lie_derivative is not None:
            plot_lie_derivative(system, net, x_test, x1, x2, ax1)

        # Subplot 2: Contour plot of target set and level sets
        ax2 = fig.add_subplot(122)

        if V_list is not None and c_lists is not None:
            for func, levels in zip(V_list, c_lists):
                func_input_split = np.split(input_array, input_array.shape[1], 
                                            axis=1)
                func_eval = func(*func_input_split).reshape(x1.shape)
                ax2.contour(x1, x2, func_eval, levels=levels,
                            colors='g', linewidths=2, linestyles='-.')    

        if c1_P is not None: 
            ax2.contour(x1, x2, quad_V_test, levels=[c1_P], 
                        colors='r', linewidths=2, linestyles='--')
        
        if c2_P is not None: 
            ax2.contour(x1, x2, quad_V_test, levels=[c2_P], 
                        colors='r', linewidths=2, linestyles='--')
        
        if c1_V is not None:
            ax2.contour(x1, x2, V_test, colors='b', levels=[c1_V])
        
        if c2_V is not None:
            cs = ax2.contour(x1, x2, V_test, colors='b', levels=[c2_V], 
                             linewidths=3)    

        if phase_portrait is not None:
            plot_phase_portrait(x1, x2, system, 
                                closed_loop_f_numpy=closed_loop_f_numpy)

        if c2_V is not None:
            ax2.clabel(cs, inline=1, fontsize=10)
        ax2.set_xlabel(r'$x_1$', fontsize=24)
        ax2.set_ylabel(r'$x_2$', fontsize=24)
        ax2.set_title('Level sets')
    lyznet.tok()    

    if obstacles is not None:
        plot_obstacles(ax2, obstacles)

    if plot_trajectories is not None:
        simulate_trajectories(system, model_path, 
                              closed_loop_f_numpy=closed_loop_f_numpy)

    if plot_cost is not None:
        plot_accumulated_cost(system, model_path, elm_model, 
                              u_func, Q, R, T=10, 
                              step_size=0.02, save_csv=True,
                              closed_loop_f_numpy=closed_loop_f_numpy)
    plt.tight_layout()
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'{model_path}.pdf', format='pdf', dpi=300)
    plt.close(fig)
    # plt.show()


def plot_V_reproduce(system, net=None, model_path=None, V_list=None, c_lists=None, 
           c1_V=None, c2_V=None, c1_P=None, c2_P=None, 
           phase_portrait=None, elm_model=None, lie_derivative=None, 
           plot_trajectories=None, n_trajectories=50, plot_cost=None,
           u_func=None, Q=None, R=None, closed_loop_f_numpy=None,
           obstacles=None):
    domain = system.domain
    d = len(system.symbolic_vars)

    print("Plotting learned Lyapunov function and level sets...")
    lyznet.tik()
    if d == 1:
        x1 = np.linspace(*domain[0], 400)
        x1 = x1.reshape(-1, 1)  # Reshape to 2D array for consistency
        input_array = x1
        P = system.P

        def quad_V(x):
            return x.T @ P @ x

        quad_V_test = np.array(
            [quad_V(x) for x in input_array]
            ).reshape(x1.shape)

        if elm_model is not None: 
            weights, bias, beta = elm_model

            def elm_V(x):
                H = np.matmul(x, weights.T) + bias.T
                return np.tanh(H) @ beta
            V_test = np.array(
                [elm_V(x) for x in input_array]
                ).reshape(x1.shape)
        elif net is not None: 
            x_test = torch.tensor(input_array, dtype=torch.float32).to(device)
            V_net = net(x_test)
            V_test = V_net.detach().cpu().numpy().reshape(x1.shape)
        else: 
            V_test = quad_V_test

        fig = plt.figure(figsize=(12, 6))  # Set figure size
        ax1 = fig.add_subplot(121)
        ax1.plot(x1, V_test, label="Learned Lyapunov Function")

        # Plotting level sets for 1D

        if c1_V is not None or c2_V is not None: 
            level_values = [c1_V, c2_V]  # Add other level values if needed
            for level in level_values:
                if level is not None:
                    level_points = x1[(V_test < level) 
                                      & (np.abs(V_test - level) < 1e-2)]
                    ax1.plot(level_points, [level] * len(level_points), 'ro')  

            for x_val in level_points:
                ax1.axvline(x=x_val, color='k', linestyle='--', alpha=0.7)

        ax1.set_xlabel(r"$x_1$", fontsize=30)
        ax1.set_ylabel("V(x)", fontsize=30)
        ax1.set_title("Learned Lyapunov Function", fontsize=30)
        ax1.legend()

    else: 
        # Generate samples for contour plot
        x1 = np.linspace(*domain[0], 200)
        x2 = np.linspace(*domain[1], 200)
        x1, x2 = np.meshgrid(x1, x2)
        # Plots are projected to (x1,x2) plane. Set other dimensions to zero.
        zero_dims = d - 2
        x_dims = [np.zeros_like(x1) for _ in range(zero_dims)]
        # Stack x1, x2, and zero dimensions to form the input tensor
        input_array = np.vstack(
            [x1.ravel(), x2.ravel()] + [x_dim.ravel() for x_dim in x_dims]
        ).T

        if system.P is not None:
            P = system.P

            def quad_V(x):
                return x.T @ P @ x

            quad_V_test = np.array([quad_V(x) for x in input_array]).reshape(x1.shape)

        if elm_model is not None: 
            weights, bias, beta = elm_model

            def elm_V(x):
                H = np.matmul(x, weights.T) + bias.T
                return np.tanh(H) @ beta
            V_test = np.array([elm_V(x) for x in input_array]).reshape(x1.shape)
        elif net is not None: 
            x_test = torch.tensor(input_array, dtype=torch.float32).to(device)
            V_net = net(x_test)
            V_test = V_net.detach().cpu().numpy().reshape(x1.shape)
        else: 
            V_test = quad_V_test

        fig = plt.figure(figsize=(12, 6))  # Set figure size

        # Subplot 1: 3D surface plot of the learned function
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_surface(x1, x2, V_test)  # Plot the learned function
        ax1.set_xlabel(r"$x_1$", fontsize=30)
        ax1.set_ylabel(r"$x_2$", fontsize=30)
        # ax1.set_zlabel("V(x)")
        ax1.set_title("Learned Lyapunov Function", fontsize=30)

        if net is not None and lie_derivative is not None:
            plot_lie_derivative(system, net, x_test, x1, x2, ax1)

        # Subplot 2: Contour plot of target set and level sets
        ax2 = fig.add_subplot(122)
        y1 = np.linspace(-3.5, 3.5, 1000)
        y2 = np.linspace(-4, 4, 1400)
        X1, X2 = np.meshgrid(y1, y2)

        V_quadratic = -1.56098 * (X1 * X2) + 2.2439 * (X1 ** 2) + 1.46341 * (X2 ** 2)

        V_6 = (-0.000272844298099 * X1 + 0.000329402792111 * X2 + 0.731965767481 * X1**2 -
            0.000515948808767 * X1**2 * X2 - 0.813284555997 * X1 * X2 + 0.599848559919 * X2**2 +
            0.507927375173 * X1**3 * X2 - 0.464110716829 * X1**2 * X2**2 + 0.000125667973758 * X1**3 +
            0.000296090140202 * X1 * X2**2 - 0.000345286233679 * X2**3 - 0.159165740963 * X1**4 +
            0.32185815208 * X1 * X2**3 - 0.141415029047 * X2**4 - 9.32638476658e-06 * X1**5 +
            0.000111487879859 * X1**4 * X2 - 0.000140761948843 * X1**3 * X2**2 + 0.00028053530111 * X1**2 * X2**3 -
            4.72613724423e-05 * X1 * X2**4 + 8.81561008609e-05 * X2**5 + 0.0115056604744 * X1**6 -
            0.0603671919674 * X1**5 * X2 + 0.122771085199 * X1**4 * X2**2 - 0.12342734661 * X1**3 * X2**3 +
            0.0729561619888 * X1**2 * X2**4 - 0.0350407146953 * X1 * X2**5 + 0.0144525984962 * X2**6)

        # Define V_nn, which can get from the pinn_van_der_pol.py
        V_nn = (0.26237118244171143 
                - 0.50079739093780518 * np.tanh(
                    -1.0827794075012207 
                    + 0.61921948194503784 * np.tanh(-1.4793460369110107 - 0.21052105724811554 * X1 - 0.27372744679450989 * X2)
                    + 0.093209430575370789 * np.tanh(-0.98550295829772949 - 0.5785408616065979 * X1 - 0.59877216815948486 * X2)
                    - 0.21036335825920105 * np.tanh(-0.94563972949981689 + 0.02181905135512352 * X1 + 0.68701893091201782 * X2)
                    - 0.42979258298873901 * np.tanh(-0.83348357677459717 - 0.8212505578994751 * X1 + 0.13336376845836639 * X2)
                    - 0.653034508228302   * np.tanh(0.017041947692632675 - 0.090094991028308868 * X1 + 0.082341447472572327 * X2)
                    - 0.27258405089378357 * np.tanh(0.78042000532150269 - 0.43576517701148987 * X1 + 0.068176165223121643 * X2)
                    - 0.46359387040138245 * np.tanh(0.83026278018951416 - 0.32804307341575623 * X1 - 1.8717272281646729 * X2)
                    - 0.13065941631793976 * np.tanh(1.0298995971679688 - 0.51479923725128174 * X1 + 0.4556383490562439 * X2)
                    + 0.28891336917877197 * np.tanh(1.450109601020813 - 1.1933362483978271 * X1 + 0.0029149351175874472 * X2)
                    + 0.23937921226024628 * np.tanh(1.6685527563095093 - 0.2964826226234436 * X1 + 1.1064096689224243 * X2)
                )
                - 0.0867595374584198 * np.tanh(
                    -0.4064633846282959 
                    + 0.096659004688262939 * np.tanh(-1.4793460369110107 - 0.21052105724811554 * X1 - 0.27372744679450989 * X2)
                    - 0.3074297308921814 * np.tanh(-0.98550295829772949 - 0.5785408616065979 * X1 - 0.59877216815948486 * X2)
                    - 0.4940744936466217 * np.tanh(-0.94563972949981689 + 0.02181905135512352 * X1 + 0.68701893091201782 * X2)
                    - 0.028568394482135773 * np.tanh(-0.83348357677459717 - 0.8212505578994751 * X1 + 0.13336376845836639 * X2)
                    - 0.32893145084381104 * np.tanh(0.017041947692632675 - 0.090094991028308868 * X1 + 0.082341447472572327 * X2)
                    + 0.51525872945785522 * np.tanh(0.78042000532150269 - 0.43576517701148987 * X1 + 0.068176165223121643 * X2)
                    + 0.12918098270893097 * np.tanh(0.83026278018951416 - 0.32804307341575623 * X1 - 1.8717272281646729 * X2)
                    + 0.1678309291601181  * np.tanh(1.0298995971679688 - 0.51479923725128174 * X1 + 0.4556383490562439 * X2)
                    + 0.31917354464530945 * np.tanh(1.450109601020813 - 1.1933362483978271 * X1 + 0.0029149351175874472 * X2)
                    - 0.087243638932704926 * np.tanh(1.6685527563095093 - 0.2964826226234436 * X1 + 1.1064096689224243 * X2)
                )
                - 0.24912264943122864 * np.tanh(
                    -0.34233087301254272 
                    - 0.23425647616386414 * np.tanh(-1.4793460369110107 - 0.21052105724811554 * X1 - 0.27372744679450989 * X2)
                    + 0.61982882022857666 * np.tanh(-0.98550295829772949 - 0.5785408616065979 * X1 - 0.59877216815948486 * X2)
                    + 0.65163666009902954 * np.tanh(-0.94563972949981689 + 0.02181905135512352 * X1 + 0.68701893091201782 * X2)
                    - 0.32302224636077881 * np.tanh(-0.83348357677459717 - 0.8212505578994751 * X1 + 0.13336376845836639 * X2)
                    + 0.23185090720653534 * np.tanh(0.017041947692632675 - 0.090094991028308868 * X1 + 0.082341447472572327 * X2)
                    - 0.23470671474933624 * np.tanh(0.78042000532150269 - 0.43576517701148987 * X1 + 0.068176165223121643 * X2)
                    - 0.36189574003219604 * np.tanh(0.83026278018951416 - 0.32804307341575623 * X1 - 1.8717272281646729 * X2)
                    + 0.5831952691078186  * np.tanh(1.0298995971679688 - 0.51479923725128174 * X1 + 0.4556383490562439 * X2)
                    + 0.64402425289154053 * np.tanh(1.450109601020813 - 1.1933362483978271 * X1 + 0.0029149351175874472 * X2)
                    + 0.71212315559387207 * np.tanh(1.6685527563095093 - 0.2964826226234436 * X1 + 1.1064096689224243 * X2)
                )
                - 0.50970333814620972 * np.tanh(
                    -0.30995723605155945 
                    + 0.2246941477060318 * np.tanh(-1.4793460369110107 - 0.21052105724811554 * X1 - 0.27372744679450989 * X2)
                    - 0.44600275158882141 * np.tanh(-0.98550295829772949 - 0.5785408616065979 * X1 - 0.59877216815948486 * X2)
                    - 1.0932924747467041 * np.tanh(-0.94563972949981689 + 0.02181905135512352 * X1 + 0.68701893091201782 * X2)
                    - 1.3093807697296143 * np.tanh(-0.83348357677459717 - 0.8212505578994751 * X1 + 0.13336376845836639 * X2)
                    - 0.49040022492408752 * np.tanh(0.017041947692632675 - 0.090094991028308868 * X1 + 0.082341447472572327 * X2)
                    - 0.37872684001922607 * np.tanh(0.78042000532150269 - 0.43576517701148987 * X1 + 0.068176165223121643 * X2)
                    + 0.072051189839839935 * np.tanh(0.83026278018951416 - 0.32804307341575623 * X1 - 1.8717272281646729 * X2)
                    + 0.26853534579277039 * np.tanh(1.0298995971679688 - 0.51479923725128174 * X1 + 0.4556383490562439 * X2)
                    - 0.46880859136581421 * np.tanh(1.450109601020813 - 1.1933362483978271 * X1 + 0.0029149351175874472 * X2)
                    + 0.39468491077423096 * np.tanh(1.6685527563095093 - 0.2964826226234436 * X1 + 1.1064096689224243 * X2)
                )
                - 0.28724205493927002 * np.tanh(
                    -0.14164865016937256 
                    + 0.74304288625717163 * np.tanh(-1.4793460369110107 - 0.21052105724811554 * X1 - 0.27372744679450989 * X2)
                    - 0.093650601804256439 * np.tanh(-0.98550295829772949 - 0.5785408616065979 * X1 - 0.59877216815948486 * X2)
                    + 0.72398912906646729 * np.tanh(-0.94563972949981689 + 0.02181905135512352 * X1 + 0.68701893091201782 * X2)
                    + 1.0409743785858154 * np.tanh(-0.83348357677459717 - 0.8212505578994751 * X1 + 0.13336376845836639 * X2)
                    + 0.23381903767585754 * np.tanh(0.017041947692632675 - 0.090094991028308868 * X1 + 0.082341447472572327 * X2)
                    + 0.16872851550579071 * np.tanh(0.78042000532150269 - 0.43576517701148987 * X1 + 0.068176165223121643 * X2)
                    + 0.12475875020027161 * np.tanh(0.83026278018951416 - 0.32804307341575623 * X1 - 1.8717272281646729 * X2)
                    + 0.30125564336776733 * np.tanh(1.0298995971679688 - 0.51479923725128174 * X1 + 0.4556383490562439 * X2)
                    + 0.47119581699371338 * np.tanh(1.450109601020813 - 1.1933362483978271 * X1 + 0.0029149351175874472 * X2)
                    - 0.14958281815052032 * np.tanh(1.6685527563095093 - 0.2964826226234436 * X1 + 1.1064096689224243 * X2)
                )
                + 0.11054860800504684 * np.tanh(
                    -0.096006877720355988 
                    + 0.67572176456451416 * np.tanh(-1.4793460369110107 - 0.21052105724811554 * X1 - 0.27372744679450989 * X2)
                    - 0.38921377062797546 * np.tanh(-0.98550295829772949 - 0.5785408616065979 * X1 - 0.59877216815948486 * X2)
                    - 1.5310935974121094 * np.tanh(-0.94563972949981689 + 0.02181905135512352 * X1 + 0.68701893091201782 * X2)
                    - 0.23653885722160339 * np.tanh(-0.83348357677459717 - 0.8212505578994751 * X1 + 0.13336376845836639 * X2)
                    - 0.53200393915176392 * np.tanh(0.017041947692632675 - 0.090094991028308868 * X1 + 0.082341447472572327 * X2)
                    + 0.59750264883041382 * np.tanh(0.78042000532150269 - 0.43576517701148987 * X1 + 0.068176165223121643 * X2)
                    + 0.57349151372909546 * np.tanh(0.83026278018951416 - 0.32804307341575623 * X1 - 1.8717272281646729 * X2)
                    + 0.61700457334518433 * np.tanh(1.0298995971679688 - 0.51479923725128174 * X1 + 0.4556383490562439 * X2)
                    + 0.44175919890403748 * np.tanh(1.450109601020813 - 1.1933362483978271 * X1 + 0.0029149351175874472 * X2)
                    - 1.554986834526062   * np.tanh(1.6685527563095093 - 0.2964826226234436 * X1 + 1.1064096689224243 * X2)
                )
                + 0.44664016366004944 * np.tanh(
                    0.099953703582286835 
                    - 0.22909188270568848 * np.tanh(-1.4793460369110107 - 0.21052105724811554 * X1 - 0.27372744679450989 * X2)
                    - 0.5247994065284729 * np.tanh(-0.98550295829772949 - 0.5785408616065979 * X1 - 0.59877216815948486 * X2)
                    + 0.49363401532173157 * np.tanh(-0.94563972949981689 + 0.02181905135512352 * X1 + 0.68701893091201782 * X2)
                    - 0.55830740928649902 * np.tanh(-0.83348357677459717 - 0.8212505578994751 * X1 + 0.13336376845836639 * X2)
                    - 0.35464167594909668 * np.tanh(0.017041947692632675 - 0.090094991028308868 * X1 + 0.082341447472572327 * X2)
                    - 0.54985737800598145 * np.tanh(0.78042000532150269 - 0.43576517701148987 * X1 + 0.068176165223121643 * X2)
                    - 0.44777429103851318 * np.tanh(0.83026278018951416 - 0.32804307341575623 * X1 - 1.8717272281646729 * X2)
                    - 0.61867767572402954 * np.tanh(1.0298995971679688 - 0.51479923725128174 * X1 + 0.4556383490562439 * X2)
                    - 0.13462585210800171 * np.tanh(1.450109601020813 - 1.1933362483978271 * X1 + 0.0029149351175874472 * X2)
                    - 1.038992166519165   * np.tanh(1.6685527563095093 - 0.2964826226234436 * X1 + 1.1064096689224243 * X2)
                )
                + 0.31676885485649109 * np.tanh(
                    0.29953297972679138 
                    + 0.11513504385948181 * np.tanh(-1.4793460369110107 - 0.21052105724811554 * X1 - 0.27372744679450989 * X2)
                    - 0.061506006866693497 * np.tanh(-0.98550295829772949 - 0.5785408616065979 * X1 - 0.59877216815948486 * X2)
                    + 0.25348550081253052 * np.tanh(-0.94563972949981689 + 0.02181905135512352 * X1 + 0.68701893091201782 * X2)
                    - 0.38706371188163757 * np.tanh(-0.83348357677459717 - 0.8212505578994751 * X1 + 0.13336376845836639 * X2)
                    - 0.064383916556835175 * np.tanh(0.017041947692632675 - 0.090094991028308868 * X1 + 0.082341447472572327 * X2)
                    + 0.10233032703399658 * np.tanh(0.78042000532150269 - 0.43576517701148987 * X1 + 0.068176165223121643 * X2)
                    - 0.23956821858882904 * np.tanh(0.83026278018951416 - 0.32804307341575623 * X1 - 1.8717272281646729 * X2)
                    + 0.088837295770645142 * np.tanh(1.0298995971679688 - 0.51479923725128174 * X1 + 0.4556383490562439 * X2)
                    + 0.36619627475738525 * np.tanh(1.450109601020813 - 1.1933362483978271 * X1 + 0.0029149351175874472 * X2)
                    + 0.70937550067901611 * np.tanh(1.6685527563095093 - 0.2964826226234436 * X1 + 1.1064096689224243 * X2)
                )
                + 0.26497375965118408 * np.tanh(
                    0.73424834012985229 
                    - 0.004535985179245472 * np.tanh(-1.4793460369110107 - 0.21052105724811554 * X1 - 0.27372744679450989 * X2)
                    + 1.3651849031448364 * np.tanh(-0.98550295829772949 - 0.5785408616065979 * X1 - 0.59877216815948486 * X2)
                    + 1.0212399959564209 * np.tanh(-0.94563972949981689 + 0.02181905135512352 * X1 + 0.68701893091201782 * X2)
                    + 1.1655539274215698 * np.tanh(-0.83348357677459717 - 0.8212505578994751 * X1 + 0.13336376845836639 * X2)
                    + 0.012068412266671658 * np.tanh(0.017041947692632675 - 0.090094991028308868 * X1 + 0.082341447472572327 * X2)
                    + 1.0592633485794067 * np.tanh(0.78042000532150269 - 0.43576517701148987 * X1 + 0.068176165223121643 * X2)
                    - 1.0534647703170776 * np.tanh(0.83026278018951416 - 0.32804307341575623 * X1 - 1.8717272281646729 * X2)
                    - 0.042276788502931595 * np.tanh(1.0298995971679688 - 0.51479923725128174 * X1 + 0.4556383490562439 * X2)
                    - 1.1845817565917969 * np.tanh(1.450109601020813 - 1.1933362483978271 * X1 + 0.0029149351175874472 * X2)
                    - 0.022373588755726814 * np.tanh(1.6685527563095093 - 0.2964826226234436 * X1 + 1.1064096689224243 * X2)
                )
                + 0.24966247379779816 * np.tanh(
                    0.86275207996368408 
                    - 0.079777687788009644 * np.tanh(-1.4793460369110107 - 0.21052105724811554 * X1 - 0.27372744679450989 * X2)
                    + 0.15378162264823914 * np.tanh(-0.98550295829772949 - 0.5785408616065979 * X1 - 0.59877216815948486 * X2)
                    - 0.74858522415161133 * np.tanh(-0.94563972949981689 + 0.02181905135512352 * X1 + 0.68701893091201782 * X2)
                    + 0.71782541275024414 * np.tanh(-0.83348357677459717 - 0.8212505578994751 * X1 + 0.13336376845836639 * X2)
                    - 0.41732850670814514 * np.tanh(0.017041947692632675 - 0.090094991028308868 * X1 + 0.082341447472572327 * X2)
                    + 0.4468209445476532 * np.tanh(0.78042000532150269 - 0.43576517701148987 * X1 + 0.068176165223121643 * X2)
                    - 0.69073379039764404 * np.tanh(0.83026278018951416 - 0.32804307341575623 * X1 - 1.8717272281646729 * X2)
                    - 0.13008217513561249 * np.tanh(1.0298995971679688 - 0.51479923725128174 * X1 + 0.4556383490562439 * X2)
                    - 0.25244593620300293 * np.tanh(1.450109601020813 - 1.1933362483978271 * X1 + 0.0029149351175874472 * X2)
                    - 0.49587270617485046 * np.tanh(1.6685527563095093 - 0.2964826226234436 * X1 + 1.1064096689224243 * X2)
                )
            )


        level_c2 = 2.3151
        level_c1 = 0.5231
        level_w2 = 0.864379
        level_w1 = 0.03295
        fixed_c1 = 0.5231
        level_SOS = 1

        cs1 = ax2.contour(X1, X2, V_quadratic, levels=[level_c1], colors='b', linewidths=1.5)
        cs2 = ax2.contour(X1, X2, V_quadratic, levels=[level_c2], colors='r', linewidths=1.5)
        cs3 = ax2.contour(X1, X2, V_nn, levels=[level_w2], colors='g', linewidths=1.5)
        cs4 = ax2.contour(X1, X2, V_nn, levels=[level_w1], colors='k', linewidths=2)
        cs5 = ax2.contour(X1, X2, V_6, levels=[level_SOS], colors='m', linewidths=2)

        h1 = cs1.collections[0]
        h2 = cs2.collections[0]
        h3 = cs3.collections[0]
        h4 = cs4.collections[0]
        h5 = cs5.collections[0]

        # ax.legend([h1, h2, h3, h4, h5],
        #           [r'$V_{quadratic}=0.5231$', r'$V_{quadratic}=2.3151$', r'$V_{NN}=0.86437$', r'$V_{NN}=0.03295$',r'$V_{SOS}=1$'],
        #           loc='best', fontsize=10)
        import matplotlib.lines as mlines

        # Create proxy artists for each contour line
        line1 = mlines.Line2D([], [], color='b', linewidth=1.5, label=r'$V_{quadratic}=0.5231$')
        line2 = mlines.Line2D([], [], color='r', linewidth=1.5, label=r'$V_{quadratic}=2.3151$')
        line3 = mlines.Line2D([], [], color='g', linewidth=1.5, label=r'$V_{NN}=0.86437$')
        line4 = mlines.Line2D([], [], color='k', linewidth=2,   label=r'$V_{NN}=0.03295$')
        line5 = mlines.Line2D([], [], color='m', linewidth=2,   label=r'$V_{SOS}=1$')

        # Add legend with the proxy artists
        ax2.legend(handles=[line1, line2, line3, line4, line5],
                loc='upper left', fontsize=12)

        ax2.set_xlabel(r'$x_1$', fontsize=30)
        ax2.set_ylabel(r'$x_2$', fontsize=30)
        ax2.set_title('Level sets', fontsize=30)
    lyznet.tok()    

    if obstacles is not None:
        plot_obstacles(ax2, obstacles)

    if plot_trajectories is not None:
        simulate_trajectories(system, model_path, 
                              closed_loop_f_numpy=closed_loop_f_numpy)

    if plot_cost is not None:
        plot_accumulated_cost(system, model_path, elm_model, 
                              u_func, Q, R, T=10, 
                              step_size=0.02, save_csv=True,
                              closed_loop_f_numpy=closed_loop_f_numpy)
    plt.tight_layout()
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'{model_path}.pdf', format='pdf', dpi=300)
    plt.close(fig)
    # plt.show()
