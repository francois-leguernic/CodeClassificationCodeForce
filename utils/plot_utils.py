import matplotlib.pyplot as plt 
from collections import Counter

def plot_tags_names_repartition(dataFrameColumn,figsize=(15,15)):
    tagCounter = Counter(tag for tags in dataFrameColumn for tag in tags)
    tagRepartition= tagCounter.most_common()
    tagNames,counts = zip(*tagRepartition)
    
    fig,ax = plt.subplots(figsize=figsize)
    ax.bar(tagNames,counts)
    ax.set_xticklabels(tagNames,rotation=45,ha="right")
    ax.set_title("Labels repartition")
    fig.tight_layout()
    plt.show()
