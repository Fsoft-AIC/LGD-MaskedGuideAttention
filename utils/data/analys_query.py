import pickle
with open('/home/tuanvovan/Documents/grasp-amodal/data/grasp-anything/prompt/0a35d0fd13323ae715e670a5e5074f9b07b8d2f4e70a1a852fbf1814d728aff6.pkl', 'rb') as f:
    x = pickle.load(f)
    prompt, queries = x

print(queries[0])