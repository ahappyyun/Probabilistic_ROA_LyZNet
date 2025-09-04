import time 
import z3
import dreal
import numpy as np
import lyznet
import sympy


def z3_global_quadratic_verifier(system, eps=1e-5):
    print('_' * 50)
    print("Verifying global stability using quadratic Lyapunov function "
          "(with Z3): ")

    z3_x = [z3.Real(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]

    dreal_x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    dreal_V = dreal.Expression(0)
    for i in range(len(dreal_x)):
        for j in range(len(dreal_x)):
            dreal_V += dreal_x[i] * system.P[i][j] * dreal_x[j]

    dreal_f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, dreal_x))
            )
        for expr in system.symbolic_f
        ]

    lie_derivative_of_V_dreal = dreal.Expression(0)
    for i in range(len(dreal_x)):
        lie_derivative_of_V_dreal += dreal_f[i] * dreal_V.Differentiate(
            dreal_x[i])
    # print(lie_derivative_of_V_dreal)

    norm_x_squared = sum([x**2 for x in z3_x])

    solver = z3.Solver()

    lie_derivative_of_V_z3 = lyznet.utils.dreal_to_z3(
        lie_derivative_of_V_dreal, z3_x)
    print("DV*f: ", lie_derivative_of_V_z3)

    solver.add(lie_derivative_of_V_z3 > -eps*norm_x_squared)
    
    result = solver.check()

    if result == z3.unsat:
        print("Verified: The EP is globally asymptotically stable.")
    else:
        print("Cannot verify global asymptotic stability. "
              "Counterexample: ")
        print(solver.model())

def z3_stochastic_global_quadratic_verifier(system, eps=1e-5):
    print('_' * 50)
    print("Verifying global stability using quadratic Lyapunov function "
          "(with Z3): ")

    z3_x = [z3.Real(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]

    dreal_x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    dreal_V = dreal.Expression(0)
    for i in range(len(dreal_x)):
        for j in range(len(dreal_x)):
            dreal_V += dreal_x[i] * system.P[i][j] * dreal_x[j]

    dreal_f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, dreal_x))
            )
        for expr in system.symbolic_f
        ]

    lie_derivative_of_V_dreal = dreal.Expression(0)
    for i in range(len(dreal_x)):
        lie_derivative_of_V_dreal += dreal_f[i] * dreal_V.Differentiate(
            dreal_x[i])
    # print(lie_derivative_of_V_dreal)

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

  
    norm_x_squared = sum([x**2 for x in z3_x])

    solver = z3.Solver()
    lie_derivative_of_V_dreal_sto = lie_derivative_of_V_dreal + trace_dreal
    lie_derivative_of_V_z3 = lyznet.utils.dreal_to_z3(
        lie_derivative_of_V_dreal_sto, z3_x)
    print("lie_derivative_of_V_z3: ", lie_derivative_of_V_z3)

    solver.add(lie_derivative_of_V_z3 > -eps*norm_x_squared)
    
    result = solver.check()

    if result == z3.unsat:
        print("Verified: The EP is globally asymptotically stable.")
    else:
        print("Cannot verify global asymptotic stability. "
              "Counterexample: ")
        print(solver.model())


def verify_quadratic_level(system, c_max=100, eps=1e-5, accuracy=1e-4):
    z3_x = [z3.Real(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]
    norm_x_squared = sum([x**2 for x in z3_x])

    xPx = sum([z3_x[i] * sum([system.P[i][j] * z3_x[j] 
               for j in range(len(z3_x))]) for i in range(len(z3_x))])

    norm_x_squared = sum([x**2 for x in z3_x])

    dreal_x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    dreal_V = dreal.Expression(0)
    for i in range(len(dreal_x)):
        for j in range(len(dreal_x)):
            dreal_V += dreal_x[i] * system.P[i][j] * dreal_x[j]

    dreal_f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, dreal_x))
            )
        for expr in system.symbolic_f
        ]

    lie_derivative_of_V_dreal = dreal.Expression(0)
    for i in range(len(dreal_x)):
        lie_derivative_of_V_dreal += dreal_f[i] * dreal_V.Differentiate(
            dreal_x[i])

    lie_derivative_of_V_z3 = lyznet.utils.dreal_to_z3(
        lie_derivative_of_V_dreal, z3_x)
 
    # print("DV*f: ", lie_derivative_of_V_z3)

    def verify_level_c(c):
        solver = z3.Solver()
        solver.add(z3.And(xPx <= c, 
                          lie_derivative_of_V_z3 + eps * norm_x_squared > 0))

        result = solver.check()
        if result == z3.unsat:
            return None
        else:
            return solver.model()

    lyznet.tik()
    c = lyznet.utils.bisection_glb(verify_level_c, 0, c_max, accuracy=accuracy)
    print(f"Region of attraction verified for x^TPx<={c}.")
    lyznet.tok()
    return c


def verify_stochastic_quadratic_level(system, c_max=100, eps=1e-5, accuracy=1e-4):
    z3_x = [z3.Real(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]
    norm_x_squared = sum([x**2 for x in z3_x])

    xPx = sum([z3_x[i] * sum([system.P[i][j] * z3_x[j] 
               for j in range(len(z3_x))]) for i in range(len(z3_x))])

    norm_x_squared = sum([x**2 for x in z3_x])

    dreal_x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    dreal_V = dreal.Expression(0)
    for i in range(len(dreal_x)):
        for j in range(len(dreal_x)):
            dreal_V += dreal_x[i] * system.P[i][j] * dreal_x[j]

    dreal_f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, dreal_x))
            )
        for expr in system.symbolic_f
        ]

    lie_derivative_of_V_dreal = dreal.Expression(0)
    for i in range(len(dreal_x)):
        lie_derivative_of_V_dreal += dreal_f[i] * dreal_V.Differentiate(
            dreal_x[i])

    # Q = np.array(system.Q).astype(np.float64)
    # r = np.min(np.linalg.eigvalsh(Q)) - eps

    # m = [
    #      dreal_f[i] - sum(system.A[i][j] * dreal_x[j] for j in range(len(dreal_x))) 
    #     for i in range(len(dreal_x))
    #     ]
    G_list = system.G

    C_columns = [sympy.Matrix(G) * sympy.Matrix(dreal_x) for G in G_list]
    C = sympy.Matrix.hstack(*C_columns)
    print(C)
    n = system.symbolic_sigma - C

    trace_term = sympy.trace(
        n.T * (system.P) * C +
        n.T * (system.P) * n +
        C.T * (system.P) * n
    )
    # print(trace_term)
    trace_dreal =lyznet.utils.sympy_to_dreal(
            trace_term, dict(zip(system.symbolic_vars, dreal_x))
            )
    
    # pm = dreal.Expression(0)
    # for i in range(len(x)):
    #     for j in range(len(m)):
    #         pm += dreal_x[i] * system.P[i][j] * m[j]
    lie_derivative_of_V_dreal = lie_derivative_of_V_dreal + trace_dreal 

    lie_derivative_of_V_z3 = lyznet.utils.dreal_to_z3(
        lie_derivative_of_V_dreal, z3_x)
 
    # print("lie_derivative_of_V_z3: ", lie_derivative_of_V_z3)

    def verify_level_c(c):
        solver = z3.Solver()
        solver.add(z3.And(xPx <= c, 
                          lie_derivative_of_V_z3 + eps * norm_x_squared > 0))

        result = solver.check()
        if result == z3.unsat:
            return None
        else:
            return solver.model()

    lyznet.tik()
    c = lyznet.utils.bisection_glb(verify_level_c, 0, c_max, accuracy=accuracy)
    print(f"Region of attraction verified for x^TPx<={c}.")
    lyznet.tok()
    return c
