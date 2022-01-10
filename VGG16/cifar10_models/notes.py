# N = 4
# level = 2**N
# partition = np.linspace(-1.2561e-07, 4.0945e-08, level)
# step=(4.0945e-08+1.2561e-07)/(level-1)
# print(partition)
# print(step)
def getpartition(N,maxval,minval):
    level = 2**N
    step=(maxval-minval)/(level-1)
    return step

def quantize(example,minval,step,N):    
    # best_match = None
    # best_match_diff = None
    # for other_val in to_values:
    #     diff = abs(other_val - val)
    #     if best_match is None or diff < best_match_diff:
    #         best_match = other_val
    #         best_match_diff = diff
    # return best_match
    level = 2**N
    I = np.around(((example.numpy()-minval)/step))     
    I[I == level] = level-1
    I[I < 0] = 0
    return (minval+I*step)

# print(res["features.0.bias"])
# example = state_dict["features.0.bias"]
# maxval = torch.max(example).item()
# minval = torch.min(example).item()
# N = 4
# step = getpartition(N, maxval, minval)
# # #print(step)
# res1 = quantize(example,minval,step,N)
# print(res1)
# level = 2**N
# # print(example)
# I = np.around(((example.numpy()-minval)/step))     
# # #print(I)
# # #print(I==level)
# I[I == level] = level-1
# I[I < 0] = 0
# print(minval+I*step)
# # m = np.arange(16)
# # print(-1.2561e-07+m*step)
# name = []
# for n, _ in state_dict.items():
#   name.append(n)
#print(name)

