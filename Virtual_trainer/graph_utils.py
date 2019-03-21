import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bokeh.io.export import get_screenshot_as_png
from bokeh.palettes import magma
from bokeh.transform import jitter
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Panel, Tabs, Slider
from bokeh.plotting import figure, save, output_file
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
from PIL import Image
import neptune


class_names = ['squat', 'deadlift', 'pushups', 'pullups', 'wallpushups', 'lunges', 'other', 'cleanandjerk']
num_of_classes = len(class_names)



def fig2pil(fig):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)

    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())




def prepare_plots(pairings,target_vals,epoch, METRICSPATH):

    cfm_file = f"{METRICSPATH}/confusion-{epoch}.png"
    bok_file = f"{METRICSPATH}/ranking_{epoch}.html"
    pairing_file = f"{METRICSPATH}/pairings_{epoch}.csv"

    classes, rankings, preds, embeds = [],[],[],[]
    idx1, idx2, dists = [], [], []

    for t_ , p_ in zip(target_vals, pairings):
        cl_,ra_,pr_,em_ = t_
        ix1_, ix2_, dist_, _ = zip(*p_)
        ix1_ = np.array(ix1_,dtype="int") + len(rankings)
        ix2_ = np.array(ix2_,dtype="int") + len(rankings)
        classes.append(cl_)
        rankings.append(ra_)
        preds.append(pr_)
        embeds.append(em_)

        idx1.append(ix1_)
        idx2.append(ix2_)
        dists.append(dist_)


    classes = np.squeeze(np.concatenate(classes))
    rankings = np.squeeze(np.concatenate(rankings))
    predictions = np.concatenate([softmax(p,axis=0) for p in preds])
    embeds = np.concatenate(embeds)
    dists = np.concatenate(dists)
    idx1 = np.concatenate(idx1)
    idx2 = np.concatenate(idx2)
    
    activations = np.argmax(predictions,axis=1) 
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
    p = figure(plot_width=600, plot_height=800, title=f"Ranking by exercise, epoch {epoch}")
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = 'Target ranking'
    p.yaxis.axis_label = 'Predicted ranking'
    

    
    for cl in range(num_of_classes):
        if cl == 6:
            continue
        df2 = df.loc[df['class']==cl]
        p.circle(x=jitter('tar', 0.5), y='pred', size=8, alpha=0.1, color=palette[cl], legend=class_names[cl], source=df2 )
        p.line(x='tar', y='pred', line_width=2, alpha=0.5, color=palette[cl], source=df2.groupby(by="tar").mean())
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    output_file(bok_file, title="Ranking by exercise")
    save(p)
    pair_df = pd.DataFrame(data={ 'idx1': idx1, 'idx2': idx2, 'dist':dists, 'true_dist':true_dist}).to_csv(pairing_file)

def prepare_plots2(pairings,target_vals,epoch, METRICSPATH):

    cfm_file = f"{METRICSPATH}/confusion-{epoch}.png"
    bok_file = f"{METRICSPATH}/ranking_{epoch}.html"
    pairing_file = f"{METRICSPATH}/pairings_{epoch}.csv"

    classes, rankings, preds, embeds = [],[],[],[]
    rank1, rank2, dists, pair_class = [], [], [], []

    for t_ , p_ in zip(target_vals, pairings):
        cl_,ra_,pr_,em_ = t_
        rk1_, rk2_, dist_, pcl_ = zip(*p_)
        classes.append(cl_)
        rankings.append(ra_)
        preds.append(pr_)
        embeds.append(em_)

        rank1.append(rk1_)
        rank2.append(rk2_)
        dists.append(dist_)
        pair_class.append(pcl_)


    classes = np.squeeze(np.concatenate(classes))
    rankings = np.squeeze(np.concatenate(rankings))
    predictions = np.concatenate([softmax(p,axis=0) for p in preds])
    embeds = np.concatenate(embeds)
    dists = np.concatenate(dists)
    rank1 = np.concatenate(rank1)
    rank2 = np.concatenate(rank2)
    pair_class = np.concatenate(pair_class)
    
    activations = np.argmax(predictions,axis=1) 
    conf_mat = confusion_matrix(classes,activations)
    fig = plt.figure(figsize=[10,8])
    plot_confusion_matrix(conf_mat, classes=class_names, normalize=False,
                      title=f'Confusion matrix epoch {epoch}')
    plt.savefig(cfm_file,format="png")
    pil_image = fig2pil(fig)
    neptune.send_image('conf_mat', pil_image)


    max_rank = np.max(rankings)
    true_dist = rank1 - rank2
    tar_d = max_rank - true_dist
    pred_d = max_rank - dists
    pair_class = pair_class.astype(int)
    df = pd.DataFrame(data={ 'tar': tar_d, 'pred': pred_d, 'class': pair_class})

    
    palette = magma(num_of_classes + 1)
    p = figure(plot_width=600, plot_height=800, title=f"Ranking by exercise, epoch {epoch}")
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = 'Target ranking'
    p.yaxis.axis_label = 'Predicted ranking'
    

    
    for cl in range(num_of_classes):
        if cl == 6:
            continue
        df2 = df.loc[df['class']==cl]
        p.circle(x=jitter('tar', 0.5), y='pred', size=8, alpha=0.1, color=palette[cl], legend=class_names[cl], source=df2 )
        p.line(x='tar', y='pred', line_width=2, alpha=0.5, color=palette[cl], source=df2.groupby(by="tar").mean())
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    output_file(bok_file, title="Ranking by exercise")
    save(p)
    pil_image2 = get_screenshot_as_png(p)
    neptune.send_image('rank_distances', pil_image2)

    pair_df = pd.DataFrame(data={ 'rank1': rank1, 'rank2': rank2, 'dist':dists, 'true_dist':true_dist}).to_csv(pairing_file)

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
 
