import hdtmm.model
import numpy as np
import sys

# Create ground-truth model
#   3 components
#   4 variables
#       1 categorical
#       1 PR (zero-inflated gamma)
#       1 ordinal
#       1 R (gaussian)

model_def = {'var1': ['C', None, None, None],
             'var2': ['PR', None, None, False],
             'var3': ['O', 1, 8, True],
             'var4': ['R', None, None, False]}

params = [[0.7, 0.2, 0.1],
          {'var1': [0.1, {'A': 0.1, 'B': 0.9}],
           'var2': [0.1, [0.2, 2, 2]],
           'var3': [0.1, [3, 1]],
           'var4': [0.1, [10, 9]]},
          {'var1': [0.2, {'A': 0.9, 'B': 0.1}],
           'var2': [0.2, [0.1, 5, 1]],
           'var3': [0.2, [6, 1]],
           'var4': [0.2, [20, 9]]},
          {'var1': [0.1, {'A': 0.5, 'B': 0.5}],
           'var2': [0.1, [0.2, 1, 2]],
           'var3': [0.1, [4, 9]],
           'var4': [0.1, [30, 100]]}]

model = hdtmm.model.hdtmm(model_def, 3)
model.num_params = 3*(2 + 4 + 3 + 3)
model.params = params

# Sample data from the model

sys.stdout.write('Sampling data ...\n')
sys.stdout.flush()
data = model.sample(10000, randstate=1)
print(data)

# Estimate a model from the data and compare parameters to the ground truth

model_est = hdtmm.model.hdtmm(model_def, 3)
model_est.estimate(data, verbose=True, th=1e-6, randstate=1)
params_est = model_est.params


idx = [-1, -1, -1]
idx[0] = np.argmin([np.abs(params[0][0] - i0) for i0 in params_est[0]])
idx[1] = np.argmin([np.abs(params[0][1] - i0) for i0 in params_est[0]])
idx[2] = np.argmin([np.abs(params[0][2] - i0) for i0 in params_est[0]])

sys.stdout.write('\n\n')
sys.stdout.write('            Truth      |   Estimated   \n')
sys.stdout.write('---------------------------------------\n')
sys.stdout.write('Pr(comp 1)  %.03f      | %.03f    \n'%(params[0][0], params_est[0][idx[0]]))
sys.stdout.write('Pr(comp 2)  %.03f      | %.03f    \n'%(params[0][1], params_est[0][idx[1]]))
sys.stdout.write('Pr(comp 3)  %.03f      | %.03f    \n'%(params[0][2], params_est[0][idx[2]]))

for c0 in range(1, 4):
    sys.stdout.write('\nComponent #%i\n'%c0)
    for v0 in params[c0]:
        p_t = params[c0][v0]
        p_e = params_est[1 + idx[c0 - 1]][v0]

        sys.stdout.write('  %s\n'%v0)
        sys.stdout.write('   p(missing)   %.03f  | %.03f\n'%(p_t[0], p_e[0]))
        if v0 == 'var1':
            for x0 in params[c0][v0][1]:
                sys.stdout.write('      %s         %.03f  | %.03f\n'%(x0, p_t[1][x0], p_e[1][x0]))
        if v0 == 'var2':
            sys.stdout.write('   Pr(zero)     %.03f  | %.03f\n'%(p_t[1][0], p_e[1][0]))
            sys.stdout.write('   k            %.03f  | %.03f\n'%(p_t[1][1], p_e[1][1]))
            sys.stdout.write('   theta        %.03f  | %.03f\n'%(p_t[1][2], p_e[1][2]))
        if v0 == 'var3':
            sys.stdout.write('   mu           %.03f  | %.03f\n'%(p_t[1][0], p_e[1][0]))
            sys.stdout.write('   sigma^2      %.03f  | %.03f\n'%(p_t[1][1], p_e[1][1]))
        if v0 == 'var4':
            sys.stdout.write('   mu          %.03f  | %.03f\n'%(p_t[1][0], p_e[1][0]))
            sys.stdout.write('   sigma^2     %.03f | %.03f\n'%(p_t[1][1], p_e[1][1]))
            
            



