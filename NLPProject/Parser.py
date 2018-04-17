import pandas as pd

comments = pd.DataFrame.from_csv('toxicity_annotated_comments.tsv', sep='\t');
tox = pd.DataFrame.from_csv('toxicity_annotations.tsv', sep='\t');
# print tox['rev_id'].count();
dfnewtox = tox.groupby('rev_id')['toxicity'].mean()
dfnewtoxscore = tox.groupby('rev_id')['toxicity_score'].mean()
df = pd.concat([comments, dfnewtox, dfnewtoxscore], axis=1)
# print comments.query('rev_id == 2232.0')
print comments.query('rev_id')
df.to_csv('out.csv',sep='\t')
