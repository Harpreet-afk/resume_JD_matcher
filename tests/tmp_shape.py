import numpy as np
import pathlib
root = pathlib.Path(r'C:\Users\hp\Documents\resume_JD_matcher')
re = np.load(root/'data'/'resume_embeddings.npy')
jd = np.load(root/'data'/'jd_embeddings.npy')
print('resume', re.shape)
print('jd', jd.shape)
