import pandas as pd

comments = pd.DataFrame.from_csv('data/toxicity_annotated_comments.tsv', sep='\t');
tox = pd.DataFrame.from_csv('data/toxicity_annotations.tsv', sep='\t');
# print tox['rev_id'].count();
dfnewtox = tox.groupby('rev_id')['toxicity'].mean()
dfnewtoxscore = tox.groupby('rev_id')['toxicity_score'].mean()
df = pd.concat([comments, dfnewtox, dfnewtoxscore], axis=1)
# print comments.query('rev_id == 2232.0')
print comments.query('rev_id')
dftrain = df.loc[df['split'] == 'train']
dftest = df.loc[df['split'] == 'test']
dfdev = df.loc[df['split'] == 'dev']

dftrain = dftrain[['comment','toxicity','toxicity_score']]
dftest = dftest[['comment','toxicity','toxicity_score']]
dfdev = dfdev[['comment','toxicity','toxicity_score']]

dftrain.to_csv('data/train.csv',sep='\t')
dftest.to_csv('data/test.csv',sep='\t')
dfdev.to_csv('data/dev.csv',sep='\t')

