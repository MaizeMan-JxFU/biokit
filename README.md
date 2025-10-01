install from github...
```bash
pip install git+https://github.com/MaizeMan-JxFU/biokit.git
```
usage of pymlm
```python
from biokit.pymlm import BLUP
geno = breader('example/maize350_9496_SNP',).iloc[:,2:].T # bfile prefix
pheno = pd.read_csv('example/pheno.tsv',sep='\t',index_col=0).iloc[:,[0]].dropna() # pheno file
print(pheno.columns)
g_p = pd.concat([geno,pheno],axis=1).dropna()
x = g_p.iloc[:,:-1].values
y = g_p.iloc[:,[-1]].values
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,shuffle=True,random_state=430)  # 初始化KFold
for i in [None,'pearson','VanRanden','gemma1','gemma2']:
    _ = []
    _hat = []
    for train_index , test_index in kf.split(x,y):
        model = BLUP(y[train_index,:],None,x[train_index,:],kinship=i) # gemma1 gemma2 VanRanden
        model.fit()
        y_hat = model.predict(None,x[test_index,:])
        _+=y[test_index,:].tolist()
        _hat+=y_hat.tolist()
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    print(f'{i}({round(model.pve,3)})',pearsonr(_,_hat).statistic**2)
plt.scatter(_,_hat)
plt.plot(_,_,color='red',linestyle='dashed')
plt.savefig('test.png')
```
