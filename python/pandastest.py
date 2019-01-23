import pandas as pd
import numpy as np

s = pd.Series([1, 2, 6, np.nan, 44, 1])

dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])

df.dtypes
df.index
df.columns

df2 == pd.DataFrame({'A':1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo'})
print(df2.values)
df2.describe()
df2.T
df2.sort_index(axis=1, ascending=False)
df2.sort_values(by'B')

df.A
df['20130102':'20130104']
df.loc['20130102']
df.loc[:,['A','B']]
df.loc['20130102',['A','B']]
df.iloc[3,1]
df.iloc[3:5,1:3]
df.iloc[[1,3,5],1:3]
df.ix[:3,['A','C']]
df[df.A>8]

df.iloc[2,2] = 1111
df.loc['20130101','B'] = 2222
df.B[df.A>4] = 0
df['F'] = np.nan
df['E'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130101',periods=6))

df.dropna(axis=0,how='any')
df.fillna(value=0)
df.isnull()

data = pd.read_csv('student.csv')
data.to_pickle('student.pickle')

res = pd.concat([df1, df2, df3], axis=0)
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

res = pd.concat([df1, df2], axis=0, join='outer')
res = pd.concat([df1, df2], axis=0, join='inner', ignore_index=True)
res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
res = df1.append(df2, ignore_index=True)
res = df1.append([df2, df3], ignore_index=True)

res = pd.merge(left, right, on='key')
res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
res = pd.merge(left, right, left_index=True, right_index=True, how='outer')

data = pd.DataFrame(
    np.random.randn(1000,4),
    index=np.arange(1000),
    columns=list("ABCD")
    )
data.cumsum()
data.plot()
plt.show()

ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')
data.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
plt.show()
