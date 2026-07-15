import pandas as pd
from pathlib import Path
root = Path(r'C:\Users\hp\Documents\resume_JD_matcher')
post = pd.read_csv(root/'data'/'raw'/'postings.csv', encoding='utf-8')
res = pd.read_csv(root/'data'/'raw'/'Resume'/'Resume.csv', encoding='utf-8')
print('postings', len(post))
print('resumes', len(res))
