import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bokeh.palettes import magma
from bokeh.transform import jitter
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Panel, Tabs, Slider
from bokeh.plotting import figure, save, output_file
from sklearn.metrics import confusion_matrix
from scipy.special import softmax


class_names = ['squat', 'deadlift', 'pushups', 'pullups', 'wallpushups', 'lunges', 'other', 'cleanandjerk']
num_of_classes = len(class_names)

def prepare_plots(pairings,target_vals,epoch, METRICSPATH):

    cfm_file = f"{METRICSPATH}/confusion-{epoch}.png"
    bok_file = f"{METRICSPATH}/ranking_{epoch}.html"
    pairing_file = f"{METRICSPATH}/pairings_{epoch}.csv"

    classes, rankings, preds, embeds = [],[],[],[]
    idx1, idx2, dists = [], [], []

    #p_ = np.concatenate(pairings)
    for t_ , p_ in zip(target_vals, pairings):
        cl_,ra_,pr_,em_ = t_
        classes.append(cl_)
        rankings.append(ra_)
        preds.append(pr_)
        embeds.append(em_)
        ix1_, ix2_, dist_, _ = p_
        ix1_ = np.array(ix1_,dtype="int") + len(rankings)
        ix2_ = np.array(ix2_,dtype="int") + len(rankings)
        idx1.append(ix1_)
        idx2.append(ix2_)
        dists.append(dist_)


    classes = np.squeeze(np.concatenate(classes))
    rankings = np.squeeze(np.concatenate(rankings))
    predictions = np.concatenate([softmax(p,axis=0) for p in preds])
    embeds = np.concatenate(embeds)
    
    
    idx1 = np.array(idx1,dtype="int")
    idx2 = np.array(idx2,dtype="int")
    dists = np.array(dists)
    
    activations = np.argmax(predictions,axis=1) 
    print(activations.shape)
    conf_mat = confusion_matrix(classes,activations)
    plt.figure(figsize=[10,8])
    plot_confusion_matrix(conf_mat, classes=class_names, normalize=False,
                      title=f'Confusion matrix epoch {epoch}')
    plt.savefig(cfm_file,format="png")

    max_rank = np.max(rankings)
    true_dist = rankings[idx1] - rankings[idx2]
    tar_d = max_rank - true_dist
    pred_d = max_rank - dists
    cl_ar = classes[idx1].astype(int)
    df = pd.DataFrame(data={ 'tar': tar_d, 'pred': pred_d, 'class': cl_ar})

    
    palette = magma(num_of_classes + 1)
    p = figure(plot_width=500, plot_height=500, title=f"Ranking by exercise, epoch {epoch}")
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = 'Target ranking'
    p.yaxis.axis_label = 'Predicted ranking'
    

    
    for cl in range(num_of_classes):
        df2 = df.loc[df['class']==cl]
        p.circle(x=jitter('tar', 0.5), y='pred', size=8, alpha=0.1, color=palette[cl], legend=class_names[cl], source=df2 )
        p.line(x='tar', y='pred', line_width=2, alpha=0.5, color=palette[cl], source=df2.groupby(by="tar").mean())
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    output_file(bok_file, title="Ranking by exercise")
    save(p)
    pair_df = pd.DataFrame(data={ 'idx1': idx1, 'idx2': idx2, 'dist':dists, 'true_dist':true_dist}).to_csv(pairing_file)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
 
