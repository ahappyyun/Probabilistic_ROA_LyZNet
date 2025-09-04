import time 
import sympy
import dreal
import lyznet


def reach_verifier_dreal(system, x, V, f, c1, c_max=100, 
                         tol=1e-4, accuracy=1e-4, number_of_jobs=32):
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain
    epsilon = 1e-4

    lie_derivative_of_V = dreal.Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V.Differentiate(x[i])

    def Check_reachability(c2):    
        x_bound = lyznet.utils.get_bound(x, xlim, V, c1_V=c1, c2_V=c2)
        x_boundary = dreal.logical_or(x[0] == xlim[0][0], x[0] == xlim[0][1])
        for i in range(1, len(x)):
            x_boundary = dreal.logical_or(x[i] == xlim[i][0], x_boundary)
            x_boundary = dreal.logical_or(x[i] == xlim[i][1], x_boundary)
        set_inclusion = dreal.logical_imply(
            x_bound, dreal.logical_not(x_boundary)
            )
        reach = dreal.logical_imply(x_bound, lie_derivative_of_V <= -epsilon)
        condition = dreal.logical_and(reach, set_inclusion)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    c_best = lyznet.utils.bisection_glb(Check_reachability, 
                                        c1, c_max, accuracy)
    return c_best

def stochastic_reach_verifier_dreal(system, x, V, lie_derivative_of_V, f, c1, c_max=100, 
                         tol=1e-4, accuracy=1e-4, number_of_jobs=64):
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain
    epsilon = 1e-4

    def Check_reachability(c2):    
        x_bound = lyznet.utils.get_bound(x, xlim, V, c1_V=c1, c2_V=c2)
        x_boundary = dreal.logical_or(x[0] == xlim[0][0], x[0] == xlim[0][1])
        for i in range(1, len(x)):
            x_boundary = dreal.logical_or(x[i] == xlim[i][0], x_boundary)
            x_boundary = dreal.logical_or(x[i] == xlim[i][1], x_boundary)
        set_inclusion = dreal.logical_imply(
            x_bound, dreal.logical_not(x_boundary)
            )
        reach = dreal.logical_imply(x_bound, lie_derivative_of_V <= -epsilon)
        condition = dreal.logical_and(reach, set_inclusion)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    c_best = lyznet.utils.bisection_glb(Check_reachability, 
                                        c1, c_max, accuracy)
    return c_best


def quadratic_reach_verifier(system, c1_P, tol=1e-4, accuracy=1e-4,
                             number_of_jobs=32):
    # Create dReal variables based on the number of symbolic variables
    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]
    print('_' * 50)
    print("Verifying ROA using quadratic Lyapunov function:")
    print("x^TPx = ", V)

    # Create dReal expressions for f based on the symbolic expressions
    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    start_time = time.time()
    c2_P = reach_verifier_dreal(system, x, V, f, c1_P, c_max=100, 
                                tol=tol, accuracy=accuracy,
                                number_of_jobs=number_of_jobs)

    if c2_P is None:
        c2_P = c1_P

    end_time = time.time()
    if c2_P > c1_P:     
        print(f"Largest level set x^T*P*x <= {c2_P} verified by reach & stay.")
    else:
        print(f"Largest level set x^T*P*x <= {c2_P} remains the same.")
    print(f"Time taken for verification: {end_time - start_time} seconds.\n")
    return c2_P


def stochastic_quadratic_reach_verifier(system, c1_P, tol=1e-4, accuracy=1e-4,
                             number_of_jobs=32):
    # Create dReal variables based on the number of symbolic variables
    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]
    print('_' * 50)
    print("Verifying ROA using quadratic Lyapunov function:")
    print("x^TPx = ", V)

    # Create dReal expressions for f based on the symbolic expressions
    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    sigma= system.symbolic_sigma
    # print(sigma)
    sigma_T_dreal = [
        [lyznet.utils.sympy_to_dreal(expr, dict(zip(system.symbolic_vars, x))) for expr in row]
        for row in sigma.T.tolist()
        ]
    # print(sigma_T_dreal)

    sigma_dreal = [
        [lyznet.utils.sympy_to_dreal(expr, dict(zip(system.symbolic_vars, x))) for expr in row]
        for row in sigma.tolist()
        ]
    # print(sigma_dreal)

    def mat_mul(A, B):
        # A  m x n, B  n x p
        m, n = len(A), len(A[0])
        p = len(B[0])
        result = [[dreal.Expression(0) for _ in range(p)] for _ in range(m)]
        for i in range(m):
            for j in range(p):
                sum_expr = dreal.Expression(0)
                for k in range(n):
                    sum_expr += A[i][k] * B[k][j]
                result[i][j] = sum_expr
        return result
    
    trace_term = mat_mul(mat_mul(sigma_T_dreal, system.P), sigma_dreal)

    # trace_term = sigma_T_dreal * V_learn_xx * sigma_dreal
    
    trace_dreal = dreal.Expression(0)
    # print(trace_term)
    for i in range(len(x)):
        trace_dreal += trace_term[i][i]

    lie_derivative_of_V = dreal.Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V.Differentiate(x[i])
    
    lie_derivative_of_V += trace_dreal

    start_time = time.time()
    c2_P = stochastic_reach_verifier_dreal(system, x, V, lie_derivative_of_V, f, c1_P, c_max=100, 
                                tol=tol, accuracy=accuracy,
                                number_of_jobs=number_of_jobs)

    if c2_P is None:
        c2_P = c1_P

    end_time = time.time()
    if c2_P > c1_P:     
        print(f"Largest level set x^T*P*x <= {c2_P} verified by reach & stay.")
    else:
        print(f"Largest level set x^T*P*x <= {c2_P} remains the same.")
    print(f"Time taken for verification: {end_time - start_time} seconds.\n")
    return c2_P
