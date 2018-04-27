import pandas as pd

comments = pd.read_table('data/toxicity_annotated_comments.tsv');
tox = pd.read_table('data/toxicity_annotations.tsv');
dfnewtox = pd.DataFrame(tox.groupby('rev_id',as_index=False)['toxicity'].mean())
dfnewtoxscore = pd.DataFrame(tox.groupby('rev_id',as_index=False)['toxicity_score'].mean())

dfmerge = pd.merge(dfnewtox,dfnewtoxscore,on='rev_id')
df=pd.merge(comments, dfmerge, on='rev_id')
# print "$$$$$$$$"
# print df.shape
# print df.tail()

dftrain = df.loc[df['split'] == 'train']
dftest = df.loc[df['split'] == 'test']
dfdev = df.loc[df['split'] == 'dev']
dftrain = dftrain[['comment','toxicity','toxicity_score']]
dftest = dftest[['comment','toxicity','toxicity_score']]
dfdev = dfdev[['comment','toxicity','toxicity_score']]
print dftrain.tail()


dftrain.to_csv('data/train.csv',sep='\t')
dftest.to_csv('data/test.csv',sep='\t')
dfdev.to_csv('data/dev.csv',sep='\t')

