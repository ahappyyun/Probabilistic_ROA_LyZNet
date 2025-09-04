import time 
import sympy
import numpy as np
import dreal 

import lyznet


def extract_dreal_Net(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net


def extract_dreal_UNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    U_net = [np.dot(h, final_layer_weight[i]) + final_layer_bias[i] 
             for i in range(final_layer_weight.shape[0])]
    # U_net = np.dot(h, final_layer_weight.T) + final_layer_bias
    return U_net


def extract_dreal_PolyNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]
    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [z[j]**2 for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net


def extract_dreal_HomoNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    norm = dreal.sqrt(sum(xi * xi for xi in x))
    h = [xi / norm for xi in x]
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    
    input_layer_weight_norm = np.linalg.norm(weights[0])

    return V_net * (norm ** model.deg), input_layer_weight_norm


def extract_dreal_HomoPolyNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    norm = dreal.sqrt(sum(xi * xi for xi in x))
    h = [xi / norm for xi in x]
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [z[j]**2 for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net * (norm ** model.deg)


def extract_dreal_SimpleNet(model, x):
    d = len(model.initial_layers)    
    weights = [layer.weight.data.cpu().numpy() 
               for layer in model.initial_layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.initial_layers]
    
    h = []
    for i in range(d):
        xi = x[i]  
        z = xi * weights[i][0, 0] + biases[i][0]  
        h_i = dreal.tanh(z) 
        h.append(h_i)
    
    final_output = sum([h_i * h_i for h_i in h])    
    return final_output


def neural_verifier(system, model, c2_P=None, c1_V=0.1, c2_V=1, 
                    tol=1e-4, accuracy=1e-2, 
                    net_type=None, number_of_jobs=60, verifier=None):
    # {x^TPx<=c2_P}: target quadratic-Lyapunov level set 
    # c1_V: target Lyapunov level set if c2_P is not specified
    # c2_V: maximal level to be verified

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    # Create dReal variables based on the number of symbolic variables
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    # print("dReal expressions of f: ", f)

    if net_type == "Simple":
        V_learn = extract_dreal_SimpleNet(model, x)
    elif net_type == "Homo": 
        V_learn, norm_W = extract_dreal_HomoNet(model, x)        
    elif net_type == "Poly":
        V_learn = extract_dreal_PolyNet(model, x)
    elif net_type == "HomoPoly":
        V_learn = extract_dreal_HomoPolyNet(model, x)
    else:
        V_learn = extract_dreal_Net(model, x)
    print("V = ", V_learn.Expand())

    lie_derivative_of_V = dreal.Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V_learn.Differentiate(x[i])

    # If homogeneous verifier is called, do the following: 
    if verifier == "Homo": 
        # config = lyznet.utils.config_dReal(number_of_jobs=32, tol=1e-7)
        norm = dreal.sqrt(sum(xi * xi for xi in x)) 
        unit_sphere = (norm == 1)
        condition_V = dreal.logical_imply(unit_sphere, V_learn >= 1e-7)
        condition_dV = dreal.logical_imply(
            unit_sphere, lie_derivative_of_V <= -1e-7
            )
        condition = dreal.logical_and(condition_V, condition_dV)
        start_time = time.time()
        result = dreal.CheckSatisfiability(
            dreal.logical_not(condition), config
            )
        if result is None:
            print("Global stability verified for homogeneous vector field!")
            # print(f"The norm of the weight matrix is: {norm_W}")
        else:
            print(result)
            print("Stability cannot be verified for homogeneous vector field!")
        end_time = time.time()
        print(f"Time taken for verifying Lyapunov function of {system.name}: " 
              f"{end_time - start_time} seconds.\n")
        return 1, 1

    quad_V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            quad_V += x[i] * system.P[i][j] * x[j]
    
    if c2_P is not None:
        target = quad_V <= c2_P

    start_time = time.time()

    def Check_inclusion(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c2_V=c1)
        condition = dreal.logical_imply(x_bound, target)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
 
    print('_' * 50)
    print("Verifying neural Lyapunov function:")

    if c2_P is not None:
        c1_V = lyznet.utils.bisection_glb(Check_inclusion, 0, 1, accuracy)
        print(f"Verified V<={c1_V} is contained in x^TPx<={c2_P}.")
    else:
        print(f"Target level set not specificed. Set it to be V<={c1_V}.")        
    c2_V = lyznet.reach_verifier_dreal(system, x, V_learn, f, c1_V, c_max=c2_V, 
                                       tol=tol, accuracy=accuracy,
                                       number_of_jobs=number_of_jobs)
    print(f"Verified V<={c2_V} will reach V<={c1_V}.")
    end_time = time.time()
    print(f"Time taken for verifying Lyapunov function of {system.name}: " 
          f"{end_time - start_time} seconds.\n")

    return c1_V, c2_V

def stochastic_neural_verifier(system, model, c2_P=None, c1_V=0.1, c2_V=1, 
                    tol=1e-4, accuracy=1e-2, 
                    net_type=None, number_of_jobs=64, verifier=None):
    # {x^TPx<=c2_P}: target quadratic-Lyapunov level set 
    # c1_V: target Lyapunov level set if c2_P is not specified
    # c2_V: maximal level to be verified

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    # Create dReal variables based on the number of symbolic variables
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    # print("dReal expressions of f: ", f)

    if net_type == "Simple":
        V_learn = extract_dreal_SimpleNet(model, x)
    elif net_type == "Homo": 
        V_learn, norm_W = extract_dreal_HomoNet(model, x)        
    elif net_type == "Poly":
        V_learn = extract_dreal_PolyNet(model, x)
    elif net_type == "HomoPoly":
        V_learn = extract_dreal_HomoPolyNet(model, x)
    else:
        V_learn = extract_dreal_Net(model, x)
    print("V = ", V_learn.Expand())

    import sympy
    def net_to_sympy_expr(model):
        # Define sympy input variables
        x_syms = sympy.symbols(f'x1:{model.layers[0].in_features + 1}')
        x = sympy.Matrix(x_syms)

        # Pass through each linear + tanh layer
        h = x
        for layer in model.layers:
            W = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            W = sympy.Matrix(W)
            b = sympy.Matrix(b)

            h = W * h + b
            h = h.applyfunc(sympy.tanh)  # apply tanh elementwise

        # Final layer (no activation)
        Wf = model.final_layer.weight.detach().cpu().numpy()
        bf = model.final_layer.bias.detach().cpu().numpy()

        Wf = sympy.Matrix(Wf)
        bf = sympy.Matrix(bf)

        V_sym = (Wf * h)[0] + bf[0]
        return V_sym

    # V_sym = sympy.sympify(V_learn.Expand())
    # Example if net is a 2-layer tanh net: V(x) = W2 * tanh(W1 * x + b1) + b2
    # x1, x2 = sympy.symbols("x1 x2")
    # x = sympy.Matrix([x1, x2])

    # W1 = model.layers[0].weight.cpu().detach().numpy()
    # b1 = model.layers[0].bias.cpu().detach().numpy()
    # W2 = model.final_layer.weight.cpu().detach().numpy()
    # b2 = model.final_layer.bias.cpu().detach().numpy()

    # h1 = sympy.Matrix(W1) * x + sympy.Matrix(b1)
    # z = h1.applyfunc(sympy.tanh)
    # V_sym = (sympy.Matrix(W2) * z)[0] + b2[0]

    V_sym = net_to_sympy_expr(model)

    # compute V_xx
    V_sym_x = [sympy.diff(V_sym, var) for var in system.symbolic_vars]
    V_sym_xx = [[sympy.diff(V_sym_x[i], var) for var in system.symbolic_vars] 
                for i in range(len(system.symbolic_vars))]

    # use sympy compute trace_term = trace( sigma^T * V_xx * sigma )
    sigma_sym = sympy.Matrix(system.symbolic_sigma)
    V_sym_xx_mat = sympy.Matrix(V_sym_xx)
    trace_term_sym = (sigma_sym.T @ V_sym_xx_mat @ sigma_sym).trace()
    # print("Symbolic trace_term = ", trace_term_sym)

    trace_dreal = lyznet.utils.sympy_to_dreal(trace_term_sym, dict(zip(system.symbolic_vars, x)))
    
    lie_derivative_of_V = dreal.Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V_learn.Differentiate(x[i])
    
    lie_derivative_of_V += 0.5*trace_dreal

    # If homogeneous verifier is called, do the following: 
    if verifier == "Homo": 
        # config = lyznet.utils.config_dReal(number_of_jobs=32, tol=1e-7)
        norm = dreal.sqrt(sum(xi * xi for xi in x)) 
        unit_sphere = (norm == 1)
        condition_V = dreal.logical_imply(unit_sphere, V_learn >= 1e-7)
        condition_dV = dreal.logical_imply(
            unit_sphere, lie_derivative_of_V <= -1e-7
            )
        condition = dreal.logical_and(condition_V, condition_dV)
        start_time = time.time()
        result = dreal.CheckSatisfiability(
            dreal.logical_not(condition), config
            )
        if result is None:
            print("Global stability verified for homogeneous vector field!")
            # print(f"The norm of the weight matrix is: {norm_W}")
        else:
            print(result)
            print("Stability cannot be verified for homogeneous vector field!")
        end_time = time.time()
        print(f"Time taken for verifying Lyapunov function of {system.name}: " 
              f"{end_time - start_time} seconds.\n")
        return 1, 1

    quad_V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            quad_V += x[i] * system.P[i][j] * x[j]
    
    if c2_P is not None:
        target = quad_V <= c2_P

    start_time = time.time()

    def Check_inclusion(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c2_V=c1)
        condition = dreal.logical_imply(x_bound, target)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
    
 
    print('_' * 50)
    print("Verifying neural Lyapunov function:")

    if c2_P is not None:
        c1_V = lyznet.utils.bisection_glb(Check_inclusion, 0, 1, accuracy)
        print(f"Verified V<={c1_V} is contained in x^TPx<={c2_P}.")
    else:
        print(f"Target level set not specificed. Set it to be V<={c1_V}.") 

    c2_V = lyznet.stochastic_reach_verifier_dreal(system, x, V_learn, lie_derivative_of_V, f, c1_V, c_max=c2_V, 
                                       tol=tol, accuracy=accuracy,
                                       number_of_jobs=number_of_jobs)
    print(f"Verified V<={c2_V} will reach V<={c1_V}.")

    def Check_inclusion_1(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c1_V=c1, c2_V=c1_V)
        # print(x_bound)
        condition1 = dreal.logical_imply(x_bound, lie_derivative_of_V <= -1e-7)
        condition2 = dreal.logical_imply(x_bound, V_learn >= 1e-7 )
        condition = dreal.logical_and(condition1, condition2)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    W_1 = lyznet.utils.bisection_lub(Check_inclusion_1, 0, c1_V, accuracy)
    print(f"Verified w<=V<={c1_V}. {W_1} is the smallest w for w<=V<={c1_V}")

    def Check_inclusion_again(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, quad_V, c2_V=c1)
        # print(x_bound)
        target_again1 = lyznet.utils.get_bound(x, xlim, V_learn, c1_V=0, c2_V=W_1)
        condition = dreal.logical_imply(target_again1, x_bound)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
    
    # target_again1 = lyznet.utils.get_bound(x, xlim, V_learn, c1_V=0, c2_V=W_1)
    # target_again2 = lyznet.utils.get_bound(x, xlim, V_learn, c1_V=0, c2_V=c1_V)
    # condition = dreal.logical_imply(target_again1, target_again2)
    # result=dreal.CheckSatisfiability(dreal.logical_not(condition), config)
    # print(result)

    C_1 = lyznet.utils.bisection_lub(Check_inclusion_again, 0, c2_P, accuracy=1e-4)
    print(f"Verified V<={W_1} is contained in x^TPx<=c. {C_1} is the smallest c for x^TPx<=c")
    
    end_time = time.time()
    print(f"Time taken for verifying Lyapunov function of {system.name}: " 
          f"{end_time - start_time} seconds.\n")

    return c1_V, c2_V, W_1, C_1
