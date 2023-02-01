import pandas
from sklearn.decomposition import PCA
import numpy
import matplotlib.pyplot as plot
import matplotlib.cm as cm

from tools import confidence_ellipse

'''
    Source data explained:
    There five pesticides whose performances are measured by three wavelengths
    Each pesticide test repeated for five times
    
    Code explained:
    Try to compress the sample data by reducing three wavelengths
    to only two PCA components and plot them.
'''

df = pandas.read_csv("data.txt", delimiter=" ")
 
fig, axs = plot.subplots(1, 2, figsize=(12, 6))

waveLenList = df.columns[1:]

dfGroup = df.groupby(by="Pesticide")
dfGroupMean = dfGroup.mean()
dfGroupStd = dfGroup.std(ddof=1)

pesticideNameList = [item for item in dfGroup.indices]
colors = cm.rainbow(numpy.linspace(0, 1, 5))

df = df.join(dfGroupMean, on="Pesticide", rsuffix="_mean")
df = df.join(dfGroupStd, on="Pesticide", rsuffix="_std")

for waveLen in waveLenList:
    df[waveLen+"_mean_all"] = df[waveLen].mean()
    df[waveLen+"_std_all"] = df[waveLen].std()

df_normalized = pandas.DataFrame()
df_normalized["Pesticide"] = df["Pesticide"]
for waveLen in waveLenList:
    #df[waveLen+"_norm"] = (df[waveLen] - df[waveLen+"_mean"]) / df[waveLen+"_std"]
    df[waveLen+"_norm"] = (df[waveLen] - df[waveLen+"_mean_all"]) / df[waveLen+"_std_all"]
    #df[waveLen+"_norm"] = df[waveLen] # no normalization
    df_normalized[waveLen+"_norm"]  = df[waveLen+"_norm"] 

### directly apply pca to all smaples
pca = PCA(n_components=df_normalized.shape[1]-1)
df_normalized_tmp = df_normalized.drop(columns=["Pesticide"])
pca.fit(df_normalized_tmp)
compressedDfTmp = df_normalized_tmp.dot( pca.components_.T)
idx = 0
for pesticide, color in zip(pesticideNameList, colors):
    
    compressedDfTmpPest = compressedDfTmp[idx*5:5+idx*5]
    
    axs[0].scatter(compressedDfTmpPest[0], compressedDfTmpPest[1], 
                  color=color, alpha=0.5, marker='o')
    axs[0].text(compressedDfTmpPest[0].mean(), compressedDfTmpPest[1].mean(), pesticide )
    
    confidence_ellipse(compressedDfTmpPest[0], compressedDfTmpPest[1], 
                        axs[0], edgecolor=color)
    
    idx += 1

### apply pca to each pesticide group
for pesticide, color in zip(pesticideNameList, colors):
    
    pca = PCA(n_components=df_normalized.shape[1]-1)
    df_normalized_tmp = df_normalized[df_normalized["Pesticide"] == pesticide].drop(columns=["Pesticide"])
    pca.fit(df_normalized_tmp)
    
    compressedDfTmp = df_normalized_tmp.dot( pca.components_.T)
    
    compressedDfTmpPest = compressedDfTmp
    
    axs[1].scatter(compressedDfTmpPest[0], compressedDfTmpPest[1], 
                  color=color, alpha=0.5, marker='o')
    axs[1].text(compressedDfTmpPest[0].mean(), compressedDfTmpPest[1].mean(), pesticide )
    
    confidence_ellipse(compressedDfTmpPest[0], compressedDfTmpPest[1], 
                       axs[1], edgecolor=color)
    
    
    
    