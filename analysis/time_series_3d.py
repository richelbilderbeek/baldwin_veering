"""
Run with python3

Args:
working_dir: contains the root directory, it expects to find a folder called "results" with the data in it
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
import pandas as pd
import argparse
import os
import sys
import glob
from multiprocessing import Pool
import functools
import operator
from matplotlib.font_manager import FontProperties # external legend
import pdb
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import itertools
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

### action codes
# 0: North
# 1: West
# 2: East
# 3: South
# 4: Eat
###

binning=25
parallel=False

def get_plot_palette(colors,n,cmap=None):
    if cmap==None:
        cmap = plt.get_cmap('cubehelix_r')
    cmap.set_bad('white',1.)
    cols=[cmap(i) for i in np.linspace(0.1,1,n)]
    if colors!=None:
        if all([isinstance(a,float) for a in colors]):
            cols=[cmap(i) for i in colors]
        elif all([isinstance(a,str) for a in colors]):
            cols=colors
        else:
            print("Warning, color palette not recognized")
    return cols

def num2str(n):
    conversion={0:"North",1:"West",2:"East",3:"South",4:"Eat"}
    return conversion[n]

def right_answer(name):
    conv={"foodH":4,            # eat
          "foodH0":4,            # eat
          "foodH1":4,            # eat
          "foodN":0,            # north
          "foodW":1,            # west
          "foodE":2,            # east
          "foodS":3,            # south
          "agentN":0,"agentW":1,"agentE":2,"agentS":3,"agentH":4
    }
    return conv[name]

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

#actions=["foodH","foodN","foodW","foodE","foodS"] # column names
#actions=["foodH","agentN","agentW","agentE","agentS"] # column names

# if __debug__:
#     print('Python version ' + sys.version)
#     print('Pandas version ' + pd.__version__)
#     print('Matplotlib version ' + matplotlib.__version__)

""" define parser: takes in three arguments, ranges = 2 args, file = 1 arg which determine the range and the output filename """
parser = argparse.ArgumentParser(description='Determine range of program and filename.')
parser.add_argument('--working_dir', metavar='working_dir', nargs=1,default="data",help='the data directory')
parser.add_argument('--heatmap_ref', metavar='heatmap_ref', nargs=1,default=[None],help='the heatmap of reference')
parser.add_argument('--season_len', metavar='season_len',type=int, nargs=1,default=False,help='the length of the seasons')
parser.add_argument('--measures', metavar='measures',nargs='?', default=False,const=True,help='if to compute the measures')
parser.add_argument('--stable_after', metavar='stable_after', nargs=1,default=0,type=int,help='the iteration after which the system is considered stable')

args = parser.parse_args()

if args:
    if args.season_len:
        binning=args.season_len[0]

def pmap(fct,lst,flag=True):
    if flag and __name__ == '__main__' and len(lst)>1:
        print("starting parallel execution")
        ans=pool.map(fct,lst)
    else:
        print("starting serial execution")
        ans=list(map(fct,lst))
    return [d for d in ans if d is not None]

def save_df(df,directory):
    """
    Saves a dataframe (or a list of) to the directory.
    """
    if not isinstance(df,list):
        df=[df]
    for i in range(len(df)):
        # df[i].columns=pd.Index([str(c) for c in df[i].columns])
        # df[i].sort_index(axis=1,inplace=True)
        df[i].to_csv(os.path.join(directory,df[i].index.names[0]+"_count.csv"),index=False)

def get_tests(files):
    """
    Returns the tests that agents performed in the current simulation.
    Extracts them from the table header
    """
    for f in files:
        try:
            ans=pd.read_csv(f,compression='infer')
        except:
            print("failed to read "+f)
            ans=None
        else:
            cols=ans.columns.values
            return [c for c in cols if c not in ['timeStep','ID','x','y','energy','action','seed','skill_0','skill_1','skill_0_gen','skill_1_gen','age']]
    return None


def widen_table(df,test):
    """
    Reshape dataframes.

    Takes as input a dataframe containing columns representing tests and containing the results (action performed) of these tests.
    Actions are coded with numbers.
    Returns a dataframe where only one test (passed as parameter) is considered and having a column for each of the actions.
    In each row only one action column contains a 1, indicating the result of the test (action chosen) for a given timestep.
    """

    df_wide=df[["timeStep","bins","ID"]].copy(deep=True) # need a deep copy to create a different object, as opposed to a view, that can have an individual identifier
    a=df.pivot(index=None,columns=test,values="value") # long to wide
    df_wide=pd.concat([df_wide,a],axis=1,copy=False)
    df_wide.index.names=[test] # identify the table
    return df_wide

def process_file(f,cols=[]):
    """
    Reads a file and adds support columns.
    """
    #print("processing file "+str(f))
    try:
        df=pd.read_csv(f,usecols=cols+['timeStep','ID','age','skill_0','skill_0_gen','energy'],compression='infer')
    except:
        print("failed to load "+f)
        return None
    else:
        df.index=df.index.astype(np.int64)
        for c in cols:
            df[c].fillna(0)         # TODO is it sensible?
            df[c]=df[c].astype(np.uint8)
        df['timeStep']=df['timeStep'].astype(np.uint32)
        df['ID']=df['ID'].astype(np.uint32)
        df['age']=df['age'].astype(np.uint32)
        df['skill_0']=df['skill_0'].astype(np.float32)
        df['skill_0_gen']=df['skill_0_gen'].astype(np.float32)
        assert(all(df["ID"]>=0))  # all IDs are valid
        df["value"]=1             # counter for aggregation
        df["value"]=df["value"].astype(np.float32)
        ## assume data is already binned
        df["bins"]=(df["timeStep"] // binning)
        df["bins"]=df["bins"].astype(np.uint32)
        return df

def process_env_file(f,cols=[]):
    """
    Reads a file and adds support columns.
    """
    #print("processing file "+str(f))
    try:
        df=pd.read_csv(f,compression='infer')
    except:
        print("failed to load "+f)
        return None
    else:
        df.index=df.index.astype(np.int64)
        df['timeStep']=df['timeStep'].astype(np.uint32)
        df['x']=df['x'].astype(np.uint16)
        df['y']=df['y'].astype(np.uint16)
        df['food0']=df['food0'].astype(np.uint32)
        df['food1']=df['food1'].astype(np.uint32)
        df['food']=df['food0']+df['food1']
        df['locked0']=df['locked0'].astype(np.uint8)
        df['locked1']=df['locked1'].astype(np.uint8)
        df['agents_in_cell']=df['agents_in_cell'].astype(np.uint16)
        df['times_unlocked0']=df['times_unlocked0'].astype(np.uint16)
        df['times_unlocked1']=df['times_unlocked1'].astype(np.uint16)
        df['times_shared0']=df['times_shared0'].astype(np.uint16)
        df['times_shared1']=df['times_shared1'].astype(np.uint16)
        df['times_shared']=df['times_shared0']+df['times_shared1']
        ## assume data is already binned
        df["bins"]=(df["timeStep"] // binning)
        df["bins"]=df["bins"].astype(np.uint32)
        return df

def process_reprod_file(f,cols=[]):
    """
    Reads a file and adds support columns.
    """
    #print("processing file "+str(f))
    try:
        df=pd.read_csv(f,usecols=cols+['timeStep','ID','parentID','skill_0','skill_0_gen'],compression='infer')
    except:
        print("failed to load "+f)
        return None
    else:
        df.index=df.index.astype(np.int64)
        df['timeStep']=df['timeStep'].astype(np.uint32)
        df['ID']=df['ID'].astype(np.uint32)
        df['parentID']=df['parentID'].astype(np.uint32)
        df['skill_0']=df['skill_0'].astype(np.float32)
        df['skill_0_gen']=df['skill_0_gen'].astype(np.float32)
        ## assume data is already binned
        df["bins"]=(df["timeStep"] // binning)
        df["bins"]=df["bins"].astype(np.uint32)
        return df

def process_forage_file(f,cols=[]):
    """
    Reads a file and adds support columns.
    """
    #print("processing file "+str(f))
    try:
        df=pd.read_csv(f,compression='infer')
    except:
        print("failed to load "+f)
        return None
    else:
        df.index=df.index.astype(np.int64)
        df['timeStep']=df['timeStep'].astype(np.uint32)
        df['ID']=df['ID'].astype(np.uint32)
        df['success']=df['success'].astype(np.uint8)
        df['food_type']=df['food_type'].astype(np.uint8)
        df['skill_0']=df['skill_0'].astype(np.float32)
        df['skill_0_gen']=df['skill_0_gen'].astype(np.float32)
        ## assume data is already binned
        df["bins"]=(df["timeStep"] // binning)
        df["bins"]=df["bins"].astype(np.uint32)
        return df

def compute_stats(data,idx=False,columns=False,drop_count=True):
    """
    Computes statistics (mean,std,confidence interval) for the given columns

    Args:
    data_: a data frame

    Kwargs:
    idx: a list of indexes on which to group, must be a list of valid column names. By default the index of the dataframe is used.
    columns: the columns to aggregate, must be a list of valide column names. By default all columns are considered

    Returns:
A data frame with columns 'X_mean', 'X_std' and 'X_ci' containing the statistics for each column name X in 'columns'
    """
    data_=data.copy()
    assert(not idx or isinstance(idx,list))
    assert(not columns or isinstance(columns,list))
    if isinstance(data_,list):
        data_=pd.concat(data_,copy=False) # join all files
    if not idx:
        idx=data_.index
        idx_c=[]
    else:
        idx_c=idx
    if not columns:
        columns=list(data_.columns[np.invert(data_.columns.isin(idx_c))])
    data_["count"]=1
    aggregations={c:[np.mean,np.std] for c in columns if c in data_._get_numeric_data().columns} # compute mean and std for each column
    aggregations.update({"count":np.sum})                # count samples in every bin
    data_=data_[columns+["count"]+idx_c].groupby(idx,as_index=False).agg(aggregations)
    # flatten hierarchy of col names
    data_.columns=["_".join(col).strip().strip("_") for col in data_.columns.values] # rename
    # compute confidence interval
    for c in columns:
        data_[c+"_ci"]=data_[c+"_std"]*1.96/np.sqrt(data_["count_sum"])
    if drop_count:
        data_.drop("count_sum",1,inplace=True)
    return data_

def avg_runs(data_):
    """
    Average data across simulations.

    Takes as input a list of dataframes, representing independent executions of one test.
    Outputs dataframe containing the average across simulations.
    """
    name=data_[0].index.names[0]
    ans=pd.concat(data_,copy=False) # concat all files together, for each test
    ans[pd.isnull(ans)]=0
    cols=[0,1,2,3,4]            # all actions
    for c in cols:
        if c not in ans.columns:
            print("Warning: action "+str(c)+" has never been performed")
            ans[c]=0
    ans=ans.rename(columns={c:str(c) for c in cols})
    ans=compute_stats(ans,idx=["bins"],columns=[str(c) for c in cols]) # group over the bins
    ans.index.names=[name]      # identify the table
    return ans

def map_widen_table(data_,tests_): return [widen_table(data_,t) for t in tests_]

def get_tests_results(tests,data_,flag=True):
    """
    Given a list of tests, it reads and aggregate all data contained in files.

    Args:
    test: a list of tests, names corresponding to columns in the data
    data: a list of dataframes to analyze

    Returns:
    A list of dataframes, each containing the average over one test
    """
    f=functools.partial(map_widen_table,tests_=tests)
    data_wide=pmap(f,data_,flag)
    data_wide=list(zip(*data_wide))
    return pmap(avg_runs,data_wide,flag)

def groupby_fct(d,index,columns,values):
    if not isinstance(index, list):
        index=[index]
    if not isinstance(columns, list):
        columns=[columns]
    if not isinstance(values, list):
        values=[values]
    return d[index+columns+values].groupby(index+columns,as_index=False)

def groupby_and_pivot_sum(d,index,columns,values):
    if values==None:
        d["count"]=1
        values="count"
    return groupby_fct(d,index,columns,values).sum().pivot(index=index,columns=columns,values=values)

def groupby_mean_fct(d,index,columns,values):
    return groupby_fct(d,index,columns,values).mean()

def groupby_sum_fct(d,index,columns,values):
    return groupby_fct(d,index,columns,values).sum()

def groupby_env(data_,index,cols,gtype=None,flag=True):
    # TODO is it sensible to combine stats like this?
    if gtype:
        if gtype=="mean":
           f=functools.partial(groupby_mean_fct,index="timeStep",columns=','.join(index),values=','.join(cols))
        if gtype=="sum":
           f=functools.partial(groupby_sum_fct,index="timeStep",columns=','.join(index),values=','.join(cols))
        data_agg=pmap(f,data_,flag) # for every agent record the skill value at every age
    else:
        data_agg=data_
    data_agg=compute_stats(data_agg,idx=index,columns=cols)
    data_agg.index.names=[str(index)+"_"+str(cols)+"_env"]
    return data_agg

def plot_measures(filename,data_,title="Evolution of parameter",season_len=False,bin_len=binning,xlim=False,col='bins',prefixes=[""]):
    x=data_[col]*bin_len
    fig=plt.figure()
    fig.suptitle(title)
    measures=["aid","ard","wid","wrd"]
    for y,i in zip(measures,range(len(measures))):
        ax = fig.add_subplot(len(measures)*100+10+i+1)
        ax.set_ylabel(y)
        ax.set_xlabel("Time")
        if xlim:
            ax.set_xlim(xlim)
        for prefix in prefixes:
            ax.plot(x,data_[prefix+y+"_mean"],label=prefix)
            ax.fill_between(x,np.asarray(data_[prefix+y+"_mean"])-np.asarray(data_[prefix+y+"_ci"]),np.asarray(data_[prefix+y+"_mean"])+np.asarray(data_[prefix+y+"_ci"]),alpha=0.2)
        if season_len:
            print(x.max())
            for xc in range(int(season_len),int(x.max()),int(season_len)):
                ax.axvline(x=xc,color='grey',linestyle='--')
        if i!=len(measures)-1:
            ax.set_xticklabels(["","","","","",""])
    if len(prefixes)>1:
        fontP = FontProperties()
        fontP.set_size('small')
        ax.legend(prop=fontP)
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def boxplot_measures(filename,data_,title="Behavior comparison",season_len=False,bin_len=binning,xlim=False,col='bins',prefixes=[""],baselines=None,prefixes_baseline=["baselo_","basehi_"],font_size=16):
    x=data_[col]*bin_len
    measures=["ard","wid"]
    labs=["Specialize","Generalize"]
    titles=["Group (ARD)","Individuals (WID)"]
    vals=[item for sublist in [[data_[prefix+m+"_mean"] for prefix in prefixes] for m in measures] for item in sublist]
    labels=[item for sublist in [[[prefix,m] for prefix in prefixes] for m in measures] for item in sublist]
    if isinstance(baselines,pd.DataFrame):
        bases=[item.mean() for sublist in [[baselines[prefix+m+"_mean"] for prefix in prefixes_baseline] for m in measures] for item in sublist]
        base_labels=[item for sublist in [[[prefix,m] for prefix in prefixes_baseline] for m in measures] for item in sublist]
    fig=plt.figure()
    boxprops = dict(linewidth=2)
    for i in range(len(measures)):
        m=measures[i]
        title=titles[i]
        ax = fig.add_subplot(100+10*len(measures)+i+1)
        ax.set_title(title,fontsize=font_size)
        ys=[vals[i] for i in range(len(vals)) if labels[i][1]==m]
        plt.boxplot(ys,boxprops=boxprops)
        if isinstance(baselines,pd.DataFrame):
            bs=[[bases[i],base_labels[i][0]] for i in range(len(vals)) if base_labels[i][1]==m]
            for y,l in bs:
                ax.axhline(y,linestyle=("-" if l=='baselo_' else "--"),linewidth=2)
        ax.set_xticklabels(["Learning","Reactive"],fontsize=font_size)
        ax.tick_params(labelsize=font_size)
        ax2=ax.twinx()
        ylim=ax.get_ylim()
        ax2.set_ylim(ylim)
        ylim_delta=[ylim[0]+abs(ylim[1]-ylim[0])*0.1,ylim[1]-abs(ylim[1]-ylim[0])*0.1]
        ax.set_yticks(ylim_delta)
        ax.set_yticklabels(labs,rotation=90,va='center')
        ax2.set_yticklabels(ax2.get_yticks(),rotation=-90,va='center')
        ax2.tick_params(labelsize=font_size)
        plt.setp(ax.get_yticklines(),visible=False)
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def plot_param_evolution(filename,data_,y,title="Evolution of parameter",xlab="X",ylab=False,ylim=[0,1.1]):
    fig,ax=plt.subplots()
    fig.suptitle(title)
    ax.set_ylabel(ylab or str(y))
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlab)
    ax.plot(data_.index,data_[y+"_mean"])
    ax.fill_between(data_.index,np.asarray(data_[y+"_mean"])-np.asarray(data_[y+"_ci"]),np.asarray(data_[y+"_mean"])+np.asarray(data_[y+"_ci"]),alpha=0.2)
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def plot_test_results(filename,data_,title="Evolution of herding behavior",answers=None,max_rng=50000,vline=False):
    """
    Plots the results of tests: every subplot contains the frequency of the 'correct answer' for a given test.

    Args:
    filename: the name of the output file
    data_: a list of dataframes

    Kwargs:
    title: the title of the plot
    answers: The 'correct answer' for each test. Defaults to None, where the correct answer is retrieved from the table's index name.
    """
    fig=plt.figure()
    fig.suptitle(title)
    for i in range(len(data_)): # save files
        # One plot for each test. For each test plot the frequency of the 'correct answer'.
        tick_interval=2000.0
        tick_marks=1000.0
        ax = fig.add_subplot(len(data_)*100+10+i+1)
        ax.set_ylim([0,1.0])
        ax.set_xlim([0,data_[i]["bins"].max()+1])
        ax.set_xlabel("")
        if vline:
            ax.axvline(vline,color="red",ls='dashed')
        if answers==None:
            correct=right_answer(data_[i].index.names[0])
        else:
            correct=answers[i]
        #Dirty hack
        if(len(data_)!=6):
            ax.set_ylabel(num2str(correct))
        else:
            if(correct==4):
                ax.set_ylabel(num2str(correct)+"-"+str(i))
            else:
                ax.set_ylabel(num2str(correct))
        ax.errorbar(data_[i]["bins"],data_[i][str(correct)+"_mean"],yerr=data_[i][str(correct)+"_ci"])
        ax.set_yticks([0.0,0.25,0.50,0.75,1.0])
        ax.set_yticklabels(["","","50%","","100%"])
        # xaxis_range=range(0,int(data_[i]["bins"].max()+1),int(tick_interval/binning))
        # xticks=[]
        # print(xticks)
        # for xval in xaxis_range:
        #     xticks.append(str(int(xval*binning/tick_marks))+"k")
        # print(xticks)
        # ax.set_xticks(xaxis_range)
        if i==len(data_)-1:
            ticks=np.unique((np.arange(0,max_rng+binning,binning)/1000).astype(np.int))
            ax.set_xticks(ticks*1000/binning)
            ax.set_xticklabels([str(t)+"k" for t in ticks])
        else:
            ax.set_xticklabels(["","","","","",""])
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def plot_action_freqs(filename,data_,actions,title="Frequency of played actions",max_rng=50000,vline=False):
    """
    Plots the frequency of all actions.

    Args:
    filename: the name of the output file
    data_: a dataframe
    actions: The actions to plot. They must match the table's column names.

    Kwargs:
    title: the title of the plot
    """
    plot_test_results(filename,[data_]*len(actions),title=title,answers=range(len(actions)),max_rng=max_rng,vline=vline)

def compute_binning(a,m,t=int,slices=None):
    """
    Args:
    a: an array of values
    m: the number of bins

    Kwargs:
    t: the type of bins. By default is int, which means slices are rounded up
    slices: the extremes of the binning intervals, useful to force the same binning on multiple datasets.

    Returns
    (slices: the extremes of the binning intervals
    ,labs: an array of the same shape as 'a' containing the bin number corresponding to each value of 'a'
    ,counts: the number of times each value in 'labs' is repeated
    )
    """
    if slices is None:
        # slices=np.arange(min(a),max(a)+step,step)
        slices=np.linspace(min(a),max(a),m)
    labs=np.digitize(a,slices)
    return slices,labs,pd.DataFrame(labs)[0].value_counts()

def data2hist(data_,col_x,col_y,weights=True,xbin_no=20,ybin_no=30,xslices=None,yslices=None):
    x=data_[col_x]
    y=data_[col_y]
    if(x.empty or y.empty):
        print("Warning, empty data with x: "+str(col_x)+" and y:"+str(col_y))
        return [None]*5
    else:
        xbin_no=min(len(x.unique()),xbin_no) # limit the number of bins in case the requested number is larger than the available points
        xslices,xidx,xcounts=compute_binning(x,xbin_no,slices=xslices)
        xbin_no=len(xslices)    # rounding errors
        yslices,yidx,ycounts=compute_binning(y,ybin_no,t=float,slices=yslices)
        if weights:
           w=1.0/xcounts[xidx] # assign to each value a weight depending on the number of occurrences (xcounts) gin that bin (xidx)
           ret,a,b=np.histogram2d(x,y,bins=[xslices,yslices],weights=w)
        else:
           ret,a,b=np.histogram2d(x,y,bins=[xslices,yslices])
        return ret,xslices,yslices,xbin_no,ybin_no

def plot_skill_evolution_heatmap(filename,data_,col_x,col_y,title="",xlab="X",xbin_no=False,binning_=1):
    if not xbin_no:
        xbin_no=len(data_[col_x].unique())
    heatmap, xlabs, ylabs,xbins,ybins=data2hist(data_,col_x,col_y,xbin_no=xbin_no)
    print(col_x,col_y,"sum per timeslice=",heatmap[0,:].sum(),heatmap[1,:].sum(),"total=",heatmap.sum())
    pd.DataFrame(heatmap).to_csv(filename+".csv",index=False)
    # plot
    plot_heatmap(filename,heatmap,xbins,ybins,xlabs*binning_,ylabs,title=title,xlab=xlab,ylab="Aptitude")

def plot_skill_evolution_existence_heatmap(filename,data_,col_x_,col_y_,title="Evolution of skill level",xlab="X",xbin_no=False,ybin_no=30,binning_=1):
    """
    converts data in histograms and plots their average
    """
    # different data might have different bins, compute a unique binning for all
    # compute binning
    x_vals=[d[col_x_].unique() for d in data_]
    x_bins=max([len(i) for i in x_vals])
    x_max=max([d[col_x_].max() for d in data_])
    x_min=min([d[col_x_].min() for d in data_])
    y_max=max([d[col_y_].max() for d in data_])
    y_min=min([d[col_y_].min() for d in data_])
    if not xbin_no:
        xbin_no=x_bins
    else:
        xbin_no=min(x_bins,xbin_no) # limit the number of bins in case the requested number is larger than the available points
    xslices=np.linspace(x_min,x_max,xbin_no)*binning_
    yslices=np.linspace(y_min,y_max,ybin_no)

    f=functools.partial(data2hist,col_x=col_x_,col_y=col_y_,weights=False,xbin_no=xbin_no,ybin_no=ybin_no,xslices=xslices,yslices=yslices)
    result=pmap(f,data_,False)
    heatmap, xlabs, ylabs,xbins,ybins=list(zip(*result))
    # check that binning is the same for all
    assert(all(list(map((lambda x: np.array_equal(x[0],x[1])),list(pairwise(xlabs))))))
    assert(all(list(map((lambda x: np.array_equal(x[0],x[1])),list(pairwise(ylabs))))))
    # check if they are all the same
    assert(len(np.unique(xbins))==1)
    assert(len(np.unique(ybins))==1)
    heatmap=np.mean(heatmap,axis=0) # average all heatmaps
    print(col_x_,col_y_,"sum per timeslice=",heatmap[0,:].sum(),heatmap[1,:].sum(),"total=",heatmap.sum())
    # plot
    plot_heatmap(filename,heatmap,xbins[0],ybins[0],xlabs[0],ylabs[0],title=title,xlab=xlab,ylab="Aptitude")

def plot_heatmap_diff(filename,data_,data_ref,col_x,col_y,title="Evolution of skill level",xlab="X"):
    heatmap, xlabs, ylabs,xbins,ybins=data2hist(data_,col_x,col_y)    # plot
    heatmap_ref, a,b,c,d=data2hist(data_ref,col_x,col_y)    # plot
    plot_heatmap(filename,(heatmap-heatmap_ref),xbins,ybins,xlabs,ylabs,title=title,xlab=xlab,ylab="Aptitude")

def plot_heatmap(filename,heatmap,xbins,ybins,xlabs,ylabs,title="Title",xlab="X",ylab="Y",font_size=16,compact_x=True):
    # yticks=np.unique(np.round(ylabs,1))
    # xticks=xlabs[0::int(len(xlabs)/min(len(xlabs),10))]*binning # take 10 of the given labels
    # xticks=[int(i) for i in xticks]
    fig=plt.figure()
    fig.suptitle(title,fontsize=font_size)
    plt.imshow(heatmap.transpose(),interpolation='none', aspect='auto', norm=matplotlib.colors.PowerNorm(gamma=0.5))
    plt.xlabel(xlab,fontsize=font_size)
    plt.ylabel(ylab,fontsize=font_size)
    if compact_x:
        xtickslabs=np.unique((xlabs/1000).astype(int))
        xtickslabs=[str(t)+"k" for t in xtickslabs]
    else:
        xtickslabs=xlabs
    nxticks=len(xtickslabs)
    nyticks=11
    plt.gca().xaxis.set_major_locator(LinearLocator(nxticks))
    plt.gca().yaxis.set_major_locator(LinearLocator(nyticks))
     # coordinates of the histogram go from 0 to ybins
    # plt.yticks(np.arange(0,ybins,ybins/float(len(yticks))))
    # plt.xticks(np.arange(0,xbins,xbins/float(len(xticks))))
    plt.gca().invert_yaxis()
    # assign previously defined labels to ticks
    plt.gca().yaxis.set_ticklabels(np.linspace(0,1,nyticks))
    plt.gca().xaxis.set_ticklabels(xtickslabs)
    plt.gca().tick_params(labelsize=font_size)
    cbar=plt.colorbar(use_gridspec=True)
    cbar.ax.tick_params(labelsize=font_size)
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def plot_histogram(filename,data_,x,title="Frequency of age",xlab="X",ylab="Y"):
    fig,ax=plt.subplots()
    fig.suptitle(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    data_[x].hist(ax=ax,bins=100)
    #ax.hist(data_["age"],bins=100)
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def filter_by_oper(df,fld,oper,thrs):
    return df[(oper(df[fld],thrs))]

def filter_data_by_oper(data_,field="timeStep",operation=operator.gt,threshold=0,flag=True):
    f=functools.partial(filter_by_oper,fld=field,oper=operation,thrs=threshold)
    data_wide=pmap(f,data_,flag) # for every agent record the skill value at every age
    return data_wide

def n_g_dot(data_,g):
    return float(data_[data_.index==g].sum(axis=1)) # sums the columns for a given row
def n_dot_s(data_,s):
    return float(data_[s].sum(axis=0)) # sums the rows for a given column
def n_dot_dot(data_):
    return float(data_.sum().sum()) # sums the rows for a given column
def p(data_,f,*args):
    """
    Usage: p(data,n_g_dot,g),p(data,n_dot_s,s)
    """
    n=f(data_,*args)
    return n/n_dot_dot(data_)
def h(data_,f,*args):
    """
    Usage: h(data,n_g_dot,g),h(data,n_dot_s,s)
    """
    x=p(data_,f,*args)
    return -(x*np.log(x))
def h_cond_g(data_,g):
    """
    p_gs/p_g.=(n_gs/n_..)/(n_g./n_..)=n_gs/n_g.
    Args:
    data_: the dataframe
    f: the function to count normalized probabilities (p_g_dot_norm or p_dot_s_norm)
    param: the parameter to f
    """
    norm=n_g_dot(data_,g)
    n=data_[data_.index==g]/norm
    return -float((n*np.log(n)).sum(axis=1))
def h_cond_s(data_,s):
    """
    p_gs/p_.s=(n_gs/n_..)/(n_.s/n_..)=n_gs/n_.s
    Args:
    data_: the dataframe
    f: the function to count normalized probabilities (p_g_dot_norm or p_dot_s_norm)
    param: the parameter to f
    """
    norm=n_dot_s(data_,s)
    n=data_[s]/norm
    return -(n*np.log(n)).sum()

def h_cond_g_weighted(data_,g):
    return h_cond_g(data_,g)*p(data_,n_g_dot,g)
def h_cond_s_weighted(data_,s):
    return h_cond_s(data_,s)*p(data_,n_dot_s,s)
def measure_ard(data_):
    """
    Is large when the group consumes the same amount of each of the S resources
    """
    return data_.columns.map(lambda s:h(data_,n_dot_s,s)).sum()
def measure_aid(data_):
    """
    Is large when each individual consumes about the same number of food items
    - e.g. when each individual generalizes across the resource types
    - or when different individuals specialize on different resources but eat about the same number of items.
    Is low when few individuals consume large amounts of food and others consume little.
    """
    return data_.index.map(lambda g:h(data_,n_g_dot,g)).sum()
def measure_wrd(data_):
    """
    Is large if total consumption of individual resources is even.
    Is low if resources are consumed unevenly, even if each individual distributes its particular level of consumption across resources in a similar proportional manner.
    """
    return data_.columns.map(lambda s:h_cond_s_weighted(data_,s)).sum()
def measure_wid(data_):
    """
    Is large if individuals generalize, and resources are consumed evenly.
    Is small if each individual selects a specialized diet.
    """
    return data_.index.map(lambda g:h_cond_g_weighted(data_,g)).sum()

def compute_entropy_measures(data_):
    """
    Computes the measures of
    - among-resource diversity (ard)
    - within-resource diversity (wrd)
    - among-individual diversity (aid)
    - within-individual diversity (wid)

    taken from the book Social foraging theory (pag 241) by Giraldeau and Caraco.

    Args:
    data_: a data frame where rows identify an individual, columns a type of resource and each entry counts the times an individual foraged a resource of that type.

    Returns: ard,aid,wrd,wid
    """
    if data_.empty:
        return {}
    ret={}
    # print("ard")
    ret.update({"ard":measure_ard(data_)})
    # print("aid")
    ret.update({"aid":measure_aid(data_)})
    # print("wrd")
    ret.update({"wrd":measure_wrd(data_)})
    # print("wid")
    ret.update({"wid":measure_wid(data_)})
    return ret

####################
# una-tantum plots #
####################

def plot_skill_surfaces_comparison(working_dir,treatments,season_len,stable_after=0,x='bins',y='skill_0',nslices=20):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Set rotation angle to 30 degrees
    ax.view_init(azim=30)
    def get_data(basedir,d,l):
        data=False
        datadir=os.path.join(basedir,d,str(l),"vis")
        if not os.path.exists(datadir):
            print("Warning: data cannot be found in dir "+str(datadir))
        else:
            data=pd.read_csv(os.path.join(datadir,"skill_0_gen_bins_heat.pdf.csv"))
        return data
    datas=[get_data(working_dir,d,season_len) for d in treatments]
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(datas)))
    for d,c in zip(datas,colors):
        x, y = np.meshgrid(d.index,d.columns.astype(int))
        d=d.T
        d[99]=d[99]/2.0
        surf = ax.plot_surface(x,y, np.asarray(d.astype(np.float32)), rstride=1, cstride=1, color=c,
                               linewidth=0, antialiased=False,label=t)
    ax.set_zlim(0.0, 0.4)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('')
    fig.savefig(os.path.join(working_dir,"surf_comp.pdf"))
    plt.close()


def plot_convergence_comparison(workingdir,treatments,season_len,ylab=None,xlab="bins",tests=["foodH0","foodH1"],colors=None,cmap=None):
    for prefix in [""]:
        fig=plt.figure()
        max_rng=0
        axs=[]
        for i in range(len(tests)):
            ax=fig.add_subplot(len(tests)*100+10+i+1)
            ax.set_ylim([0,1])
            ax.set_xlabel("Time")
            ax.set_yticklabels(["0%","","","","","100%"])
            ax.set_xticklabels(["","","","","",""])
            axs.append(ax)
        #fig.suptitle(title)
        print("TEST: ",len(tests))
        #pdb.set_trace()
        for t in treatments:
            files=glob.glob(os.path.join(workingdir,t,str(season_len),"results",prefix+"stats_agents_*.csv.bz2"))
            print("results",prefix+"stats_agents_*.csv.bz2")
            if len(files)==0:
                print("Warning: agents file list is empty. Aborting.")
            else:
                #tests=get_tests(files)
                f=functools.partial(process_file,cols=tests+["action"])
                data=pmap(f,files,parallel) # read all files
                max_rng=max(max_rng,max([d['timeStep'].max() for d in data]))
                ## Plot test results
                data_temp=get_tests_results(tests,data,parallel)
                print("DATA_TEMP: ",len(data_temp))
                for d,i,c in zip(data_temp,range(len(data_temp)),get_plot_palette(colors,len(data_temp),cmap)):
                    correct=right_answer(d.index.names[0])
                    #ax.errorbar(d["bins"],d[str(correct)+"_mean"],yerr=d[str(correct)+"_ci"],label=t)
                    print(i)
                    print(len(data_temp))
                    axs[i].set_ylabel(d.index.names[0])
                    axs[i].plot(d["bins"],d[str(correct)+"_mean"],label=t,color=c)
                    axs[i].fill_between(d["bins"],np.asarray(d[str(correct)+"_mean"])-np.asarray(d[str(correct)+"_ci"]),np.asarray(d[str(correct)+"_mean"])+np.asarray(d[str(correct)+"_ci"]),alpha=0.2,color=c)
        # add ticks to last plot
        ticks=np.unique((np.arange(0,max_rng+binning,binning)/1000).astype(np.int))
        axs[-1].set_xticks(ticks*1000/binning)
        axs[-1].set_xticklabels([str(t)+"k" for t in ticks])
        axs[-1].legend()
        fig.savefig(os.path.join("/cluster/work/gess/coss/projects/momentum/repo_leonel/momentum",prefix+"plot_convergence.pdf"),format='pdf')
        plt.close(fig)

def plot_genome_comparison(workingdir,treatment1,treatment2,season_len,col_y="skill_0_gen_mean",col_x="bins",bins=[1,3,19]):
    c1=pd.read_csv(os.path.join(workingdir,treatment1,str(season_len),"skill_0_gen_bins_count.csv"))
    c2=pd.read_csv(os.path.join(workingdir,treatment2,str(season_len),"skill_0_gen_bins_count.csv"))
    c1h, c1xlabs, c1ylabs,c1xbins,c1ybins=data2hist(c1,col_x,col_y)
    c2h, c2xlabs, c2ylabs,c2xbins,c2ybins=data2hist(c2,col_x,col_y)
    filename=os.path.join(workingdir,"genome_comparison.pdf")
    fig=plt.figure()
    for i in range(len(bins)):
        b=bins[i]
        ax = fig.add_subplot(len(bins)*100+10+i+1)
        ax.set_ylabel("bin"+str(b))
        ax.plot(c1ylabs[1:],c1h[b],label="learn")
        ax.plot(c2ylabs[1:],c2h[b],label="nolearn")
    ax.set_xlabel(col_y)
    ax.legend()
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def plot_skill_final_conf_comparison(workingdir,dirs,labs,season_len,font_size=16,title="",colors=None,cmap=None):
    def get_data(basedir,d,l):
        data=False
        datadir=os.path.join(basedir,d,str(l),"vis")
        if not os.path.exists(datadir):
            print("Warning: data cannot be found in dir "+str(datadir))
        else:
            data=pd.read_csv(os.path.join(datadir,"skill_0_gen_bins_heat.pdf.csv"))
        return data
    datas=[get_data(workingdir,d,season_len) for d in dirs]
    fig,ax=plt.subplots()
    fig.suptitle(title)
    ax.set_xlabel("Aptitude",fontsize=font_size)
    ax.set_ylabel("Frequency",fontsize=font_size)
    ax.tick_params(labelsize=font_size)
    for d,l,c in zip(datas,labs,get_plot_palette(colors,len(labs),cmap)):
        ln=20
        tmp=d[-ln:]
        ax.plot(np.asarray(tmp.columns.astype(int)),tmp.mean(),label=l,linewidth=2,color=c)
        ax.fill_between(np.asarray(tmp.columns.astype(int)),tmp.mean()+tmp.std(),tmp.mean()-tmp.std(),alpha=0.2,color=c)
        # ax.plot(np.asarray(d[-1:])[0],label=l)
        nxticks=11
        ax.set_xticklabels(np.linspace(0,1,nxticks))
        ax.xaxis.set_major_locator(LinearLocator(nxticks))
    ax.legend(fontsize=font_size)
    plt.tight_layout(pad=1.5)
    fig.savefig(os.path.join(workingdir,"skill_comparison_final.pdf"))
    plt.close(fig)

def plot_skill_final_conf_comparison_stacked(workingdir,dirs,labs,season_len,font_size=16,title="",colors=None,cmap=None):
    def get_data(basedir,d,l):
        data=False
        datadir=os.path.join(basedir,d,str(l),"vis")
        if not os.path.exists(datadir):
            print("Warning: data cannot be found in dir "+str(datadir))
        else:
            data=pd.read_csv(os.path.join(datadir,"skill_0_gen_bins_heat.pdf.csv"))
        return data
    datas=[get_data(workingdir,d,season_len) for d in dirs]
    f, axes = plt.subplots(1,2, sharey=True, sharex=True)
    nxticks=5
    ln=5
    ampfact=500
    sim_length=5000
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Time",fontsize=font_size)
    axes[0].set_xticklabels(np.linspace(0,1,nxticks))
    axes[0].xaxis.set_major_locator(LinearLocator(nxticks))
    nyticks=6
    ticks=np.linspace(0,100,nyticks)

    sim_length=sim_length/1000
    axes[0].set_yticks(np.linspace(0,100,nyticks))
    axes[0].set_yticklabels([str(int((t/100)*sim_length))+"k" for t in ticks])
    axes[0].yaxis.set_label_coords(-0.15,0.15)


    #Measure on plot arrow

    perc=30 #percentage
    lstartx=6 #location x
    lstarty=-50 #-100 location y
    uendy=lstarty-(perc/100)*ampfact


    axes[0].annotate("",
            xy=(lstartx, lstarty), xycoords='data',
            xytext=(lstartx, uendy), textcoords='data',
            arrowprops=dict(arrowstyle="|-|",
                            connectionstyle="arc3"),
            )
    axes[0].annotate(str(perc)+"%",
            xy=(lstartx, lstarty), xycoords='data',
            xytext=(lstartx-2, uendy), textcoords='data'
            )

    #Frequency distributions
    for d,l,c,ax in zip(datas,labs,get_plot_palette(colors,len(labs),cmap),axes):
        ax.set_title(l)
        ax.tick_params(labelsize=font_size)
        ax.set_xlabel("Aptitude",fontsize=font_size)
        end=len(d.index)
        for start in range(0,end,ln):
            tmp=d[start:start+ln]
            offset=start
            alph=start/end+0.1
            if start == end-ln:
                alph=1
            #print("ALPHA ",end,start,end/(start+1))
            ax.axhline(start, linestyle='--', color="grey",alpha=alph*0.5) # horizontal lines
            ax.plot(np.asarray(tmp.columns.astype(int)),(-1*tmp.mean())*ampfact+offset,label=l,linewidth=2,color=c,alpha=alph)
            #ax.fill_between(np.asarray(tmp.columns.astype(int)),(-1*tmp.mean()+tmp.std())*ampfact+offset,(-1*tmp.mean()-tmp.std())*ampfact+offset,alpha=0.2,color=c)
            # ax.plot(np.asarray(d[-1:])[0],label=l)
    #plt.tight_layout(pad=1.5)
    f.savefig(os.path.join(workingdir,"skill_comparison_history.pdf"))
    plt.close(f)
    #f.clear()


def compute_avg_foraging(data_):
    asd=data_.copy()
    # get the skill level for each agent. It does not change during lifetime, so we can average
    skills=asd[["ID","skill_0"]].groupby("ID",as_index=False).mean()
    # count how many foraging actions occurred for each food type
    asd["value"]=1
    asd=groupby_sum_fct(asd,"ID","food_type","value")
    asd=asd.merge(skills,on="ID",how="outer")
    # normalize
    def norm_val(r):
        return r["value"]/asd[asd["ID"]==r["ID"]].sum()["value"]
    asd["value_norm"]=asd.apply(norm_val,axis=1) # compute percentage of foraging both food types
    asd=asd[asd["food_type"]==0] # drop foraging food 1
    # compute stats over all agents
    asd['one']=1            # new index, we only need one average
    return compute_stats(asd,idx=['one'],columns=["value_norm","skill_0"])

def compute_specializ(workingdir):
    # load data
    datadir=os.path.join(workingdir,"results")
    if not os.path.exists(datadir):
        print("Warning: data cannot be found")
        exit(1)
    outdir=os.path.join(workingdir,"vis")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    files=glob.glob(os.path.join(datadir,"stats_forage_*.csv.bz2"))
    if len(files)==0:
        print("Warning: reprod file list is empty. Aborting.")
    else:
        data=pmap(process_forage_file,files,parallel) # read all files
        data_agg=pd.concat(data)
        # compute average
        avg_int=compute_avg_foraging(data_agg)
        # reduce data to only successful forage
        data_agg_effective=data_agg[data_agg["success"]==1]
        avg_eff=compute_avg_foraging(data_agg_effective)
        print("specialization is "+str(avg_int["value_norm_mean"][0])+" for intentions and "+str(avg_eff["value_norm_mean"][0])+" effective")
        return avg_int,avg_eff

def compute_expediting_comparison(workingdir,dir_base,dir_slow,dir_quick,font_size=16,title="",colors=None,cmap=None,season_len=None):
    # load data
    def get_data(basedir,d):
        data=False
        datadir=os.path.join(workingdir,d)
        if not os.path.exists(datadir):
            print("Warning: data cannot be found in dir "+str(datadir))
        else:
            data=pd.read_csv(os.path.join(datadir,"skill_0_gen_bins_behavior_count.csv"))
        return data
    base,slow,fast=[get_data(workingdir,d) for d in [dir_base,dir_slow,dir_quick]]
    fig,ax=plt.subplots()
    fig.suptitle(title,fontsize=font_size)
    ax.set_ylabel("Aptitude",fontsize=font_size)
    ax.set_xlabel("Time",fontsize=font_size)
    ax.tick_params(labelsize=font_size)
    for d,lab,c in zip([base,slow,fast],["Baseline","Speed up","Slow down"],get_plot_palette(colors,3,cmap)):
        ax.plot(d["bins"]*binning,d["skill_0_gen_mean"],label=lab,linewidth=2,color=c)
        ax.fill_between(d["bins"]*binning,d["skill_0_gen_mean"]-d["skill_0_gen_ci"],d["skill_0_gen_mean"]+d["skill_0_gen_ci"],alpha=0.2,color=c)
    ax.legend(fontsize=font_size)
    if season_len!=None:
        ax.axvline(season_len,color='grey',linestyle='dashed')
    filename=os.path.join(workingdir,"expediting_comparison.pdf")
    plt.tight_layout(pad=1.5)
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def compare_measures(basedir,dir1,dir2,baseline_lo_dir,baseline_hi_dir,suffix,season_len,x="bins",bin_len=binning,xlim=False,name1="1",name2="2",normalize=True):
    def get_datadir(basedir,d):
        res=os.path.join(basedir,d,"vis")
        if not os.path.exists(res):
            print("Warning: data cannot be found in dir "+str(res))
            res=False
        return res
    def read_measures(dname,suffix,x='bins'):
        measures=pd.DataFrame()
        if os.path.isfile(os.path.join(dname,"measures_"+x+"_"+suffix+".csv")):
            measures=pd.read_csv(os.path.join(dname,"measures_"+x+"_"+suffix+".csv"))
        else:
            print("Measures not found in "+str(dname))
        if normalize:
            ## pairs of measures should sum up to the same number
            norm1=measures['ard']+measures['wrd']
            norm2=measures['aid']+measures['wid']
            assert(max(norm1-norm2)<0.001)
            #measures.drop("Unnamed: 0",axis=1,inplace=True)
            ## normalize measures
            measures['ard']=measures['ard']/norm1
            measures['wrd']=measures['wrd']/norm1
            measures['aid']=measures['aid']/norm2
            measures['wid']=measures['wid']/norm2
        measures=compute_stats(measures,idx=[x])
        return measures
    # dirs
    outdir=os.path.join(basedir,"measures_compare")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    measures_1,measures_2,base_1,base_2=[read_measures(d,suffix) if d else False for d in [get_datadir(basedir,d) for d in [dir1,dir2,baseline_lo_dir,baseline_hi_dir]]]
    if not all([isinstance(i, pd.DataFrame) for i in [measures_1,measures_2,base_1,base_2]]):
        print("Some data not found, terminating.")
    else:
        # merge tables
        measures_1.columns=[c if c == 'bins' else name1+"_"+c for c in measures_1.columns]
        measures_2.columns=[c if c == 'bins' else name2+"_"+c for c in measures_2.columns]
        base_1.columns=[c if c == 'bins' else "baselo_"+c for c in base_1.columns]
        base_2.columns=[c if c == 'bins' else "basehi_"+c for c in base_2.columns]
        measures=measures_1.merge(measures_2,on=x,how='outer')
        plot_measures(os.path.join(outdir,"measures_"+x+"_"+suffix+".pdf"),measures,bin_len=bin_len,xlim=xlim,season_len=season_len,col=x,prefixes=[name1+"_",name2+"_"],title="Behavior of genotype and phenotype")
        baselines=base_1.merge(base_2,on=x,how='outer')
        boxplot_measures(os.path.join(outdir,"measures_"+x+"_"+suffix+"_boxplot.pdf"),measures,bin_len=bin_len,xlim=xlim,season_len=season_len,col=x,prefixes=[name1+"_",name2+"_"],baselines=baselines)

def compute_measure_binned(data_,suffix="",xlim=False,x='bins',bin_len=binning,season_len=1000):
    measures=pd.DataFrame()
    if os.path.isfile(os.path.join(outdir,"measures_"+x+"_"+suffix+".csv")):
        measures=pd.read_csv(os.path.join(outdir,"measures_"+x+"_"+suffix+".csv"))
    else:
        rng=range(xlim[0]//bin_len,xlim[1]//bin_len+1) if xlim else range(max([df[x].max() for df in data_]))
        for i in rng:
            print("computing measures for "+x+" "+str(i))
            f=functools.partial(groupby_and_pivot_sum,index="ID",columns="food_type",values=None)
            data_pivoted=pmap(lambda df:f(df.loc[df[x]==i]),data_,parallel) # for every agent record the skill value at every age
            dicts=pmap(compute_entropy_measures,data_pivoted,parallel) # compute measures for each simulation
            res={}
            for l in dicts:
                for k,v in l.items():
                    res.setdefault(k,[]).append(v)  # merge results
            res=pd.DataFrame(res)
            res[x]=i
            measures=measures.append(res,ignore_index=True)
            measures.to_csv(os.path.join(outdir,"measures_"+x+"_"+suffix+".csv"),index=False)
    measures=compute_stats(measures,idx=[x])
    plot_measures(os.path.join(outdir,"measures_"+x+"_"+suffix+".pdf"),measures,bin_len=bin_len,xlim=xlim,season_len=season_len,col=x)

def plot_foraging_freq_hist(workingdir,dirs,labs,bin_no=31,title="",xlab="Foraging history",ylab="Frequency",font_size=16,styles=None,colors=None,cmap=None):
    def compute_freqs(data,bin_no):
        res=pd.DataFrame()
        res_cnt=pd.DataFrame()
        for d,i in zip(data,range(len(data))):
            ## compute the frequency at which each agent forages each food type
            asd=compute_stats(d,idx=['ID','food_type'],drop_count=False)[['ID','food_type','count_sum']]
            normaliz=asd.groupby('ID',as_index=False).sum()[['ID','count_sum']]
            normaliz.rename(columns={'count_sum':'norm'},inplace=True)
            asd=pd.merge(asd,normaliz,on='ID')
            asd['count_sum']/=asd['norm']
            asd.rename(columns={'count_sum':'freq'},inplace=True)
            ## bin the data
            asd=asd[asd['food_type']==1]
            slices,labs,counts=compute_binning(asd['freq'],bin_no,t=float,slices=np.linspace(0,1,bin_no))
            asd['labs']=labs
            asd['rep']=i
             # deal with missing values
            counts=pd.DataFrame(counts,columns=['freq'],index=range(bin_no))
            counts=counts.fillna(0)
            counts['labs']=counts.index
            counts['freq']/=counts['freq'].sum()
            if res.empty:
                res=asd
            else:
                res=res.append(asd)
            if res_cnt.empty:
                res_cnt=counts
            else:
                res_cnt=res_cnt.append(counts)
        return res,res_cnt
    types=["intention","effective"]
    hist_intention=[]
    hist_effective=[]
    save_data=True
    if all([os.path.isfile(os.path.join(workingdir,"hist_foraging_"+t+"_l_"+str(l)+".csv"))for t,l in itertools.product(types,labs)]): # files exist
        print("Reading pre-saved data")
        save_data=False
        hist_intention=[pd.read_csv(os.path.join(workingdir,"hist_foraging_intention_l_"+str(l)+".csv")) for l in labs]
        hist_effective=[pd.read_csv(os.path.join(workingdir,"hist_foraging_effective_l_"+str(l)+".csv")) for l in labs]
    else:                       # generate data from sim output
        for dr in dirs:
            files=glob.glob(os.path.join(workingdir,dr,"results","stats_forage_*.csv.bz2"))
            if len(files)==0:
                print("Warning: env file list is empty. Aborting.")
            else:
                data=pmap(process_forage_file,files,parallel) # read all files
                ## remove simulations that did not complete
                sim_len=np.nanmax([d['timeStep'].max() for d in data])
                data=[d for d in data if d['timeStep'].max()==sim_len]
                ## compute foraging frequencies
                res_intention,counts_i=compute_freqs(data,bin_no)
                counts_i=compute_stats(counts_i,idx=['labs'])
                hist_intention.append(counts_i)
                data_eff=[d[d['success']==1] for d in data]
                res_effective,counts_e=compute_freqs(data_eff,bin_no)
                counts_e=compute_stats(counts_e,idx=['labs'])
                hist_effective.append(counts_e)
    for df,n in zip([hist_intention,hist_effective],types):
        fig,ax=plt.subplots()
        fig.suptitle(title,fontsize=font_size)
        ax.set_xlabel(xlab,fontsize=font_size)
        ax.set_xlim([1,bin_no-1])
        ax.set_ylabel(ylab,fontsize=font_size)
        ax.tick_params(labelsize=font_size)
        if styles==None:
            styles=['solid']*len(labs)
        for d,l,s,c in zip(df,labs,styles,get_plot_palette(colors,len(labs),cmap)):
            ax.plot(d['labs'],d['freq_mean'],label=l,linewidth=2,linestyle=s,color=c)
            ax.fill_between(d['labs'],d['freq_mean']-d['freq_ci'],d['freq_mean']+d['freq_ci'],alpha=0.2,color=c)
            if save_data:
                d.to_csv(os.path.join(workingdir,"hist_foraging_"+n+"_l_"+l+".csv"),index=False)
        ax.legend(fontsize=font_size)
        ax.xaxis.set_major_locator(LinearLocator(11))
        ax.set_xticklabels(np.linspace(0,1,11))
        fig.savefig(os.path.join(workingdir,"hist_foraging_"+n+".pdf"),format='pdf')
        plt.tight_layout(pad=1.5)
        plt.close(fig)

if __name__=='__main__':
    pool=Pool()

    workingdir=args.working_dir[0]
    print("Processing directory: "+str(os.path.abspath(workingdir)))
    heatmap_ref_dir=args.heatmap_ref[0]
    if not os.path.exists(workingdir):
        print("Warning: directory cannot be found")
        exit(1)
    datadir=os.path.join(workingdir,"results")
    if not os.path.exists(datadir):
        print("Warning: data cannot be found")
        exit(1)
    outdir=os.path.join(workingdir,"vis")
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    ####################
    # read agent files #
    ####################
    for prefix in ["phen_",""]:
        print("Processing agents files")
        files=glob.glob(os.path.join(datadir,prefix+"stats_agents_*.csv.bz2"))
        if len(files)==0:
            print("Warning: agents file list is empty. Aborting.")
        else:
            tests=get_tests(files)
            f=functools.partial(process_file,cols=tests+["action"])
            data=pmap(f,files,parallel) # read all files
            max_rng=max([d['timeStep'].max() for d in data])
            ## Plot test results
            data_temp=get_tests_results(tests,data,parallel)
            #save_df(data_temp,workingdir)
            plot_test_results(os.path.join(outdir,prefix+"action_freq.pdf"),data_temp,max_rng=max_rng,vline=False)

            ## plot frequency of actions played by agents
            tests=["action"]
            data_temp=get_tests_results(tests,data,parallel)
            #save_df(data_temp,workingdir)
            plot_action_freqs(os.path.join(outdir,prefix+"play_freq.pdf"),data_temp[0],[0,1,2,3,4],max_rng=max_rng,vline=False)

    if (int(args.stable_after[0]) if isinstance(args.stable_after,list) else int(args.stable_after))>0:
        data_filtered=filter_data_by_oper(data,"timeStep",operator.gt,args.stable_after,parallel)
    else:
        data_filtered=data
    data_filtered=[d for d in data_filtered if not d.empty]
    ## remove simulations that did not complete
    sim_len=max([d['timeStep'].max() for d in data_filtered])
    data_filtered=[d for d in data_filtered if d['timeStep'].max()==sim_len]
    print("Keeping "+str(len(data_filtered))+" simulations")
    if not data_filtered==[]:
        for y in ["skill_0","skill_0_gen"]:
            for x,xlab in zip(["age","bins"],["Age","Time"]):
                #plot existence heatmap
                binning_=(binning if x=="bins" else 1)
                for data_f in data_filtered:
                    data_f.index.names=[str(y)+"_"+str(x)] # give a name to every table
                plot_skill_evolution_existence_heatmap(os.path.join(outdir,y+"_"+x+"_existence_heat.pdf"),data_filtered,x,y,xlab=xlab,xbin_no=50,binning_=binning_)
                ## plot heatmap
                data_temp=compute_stats(data_filtered,idx=["ID",x],columns=[y]) # aggregate them
                data_temp.index.names=[str(y)+"_"+str(x)]
                if heatmap_ref_dir:
                    try:
                        data_ref=pd.read_csv(os.path.join(heatmap_ref_dir,y+"_"+x+"_count.csv"),compression='infer')
                        plot_heatmap_diff(os.path.join(outdir,y+"_"+x+"_heat_diff.pdf"),data_temp,data_ref,x,y+"_mean",xlab=xlab)
                    except:
                        print("failed to load "+os.path.join(heatmap_ref_dir,y+"_"+x+"_count.csv"))
                #save_df(data_temp,workingdir)
                plot_skill_evolution_heatmap(os.path.join(outdir,y+"_"+x+"_heat.pdf"),data_temp,x,y+"_mean",xlab=xlab,binning_=binning_)
                #we should do this only once
                if x=="age" and y=="skill_0":
                    plot_histogram(os.path.join(outdir,"age_hist.pdf"),data_temp,x,xlab=xlab,ylab="Count")
                ## plot behavior
                data_temp=compute_stats(data_filtered,idx=[x],columns=[y]) # aggregate them
                data_temp.index.names=[str(y)+"_"+str(x)+"_behavior"]
                save_df(data_temp,workingdir)
                plot_param_evolution(os.path.join(outdir,y+"_"+x+".pdf"),data_temp,y,xlab=xlab)
        # plot average population energy
        y="energy"
        for x,xlab in zip(["age","bins"],["Age","Time"]):
            data_temp=groupby_env(data_filtered,[x],[y],"mean",parallel)
            save_df(data_temp,workingdir)
            plot_param_evolution(os.path.join(outdir,y+"_"+x+".pdf"),data_temp,y,xlab=xlab,title="Average "+y+" of population",ylim=False)
    ##################
    # read env files #
    ##################
    print("Processing environment files")
    files=glob.glob(os.path.join(datadir,"stats_env_*.csv.bz2"))
    if len(files)==0:
        print("Warning: env file list is empty. Aborting.")
    else:
        data=pmap(process_env_file,files,parallel) # read all files
        y="proportion" #can be separated by food type
        title="Proportion of food0"
        f=functools.partial(groupby_sum_fct,index="timeStep",columns='bins',values=['food0','food1'])
        data_temp=pmap(f,data,parallel)
        data_temp=pd.concat(data_temp)
        data_temp['proportion']=data_temp['food0']/(data_temp['food0']+data_temp['food1'])
        data_temp=compute_stats(data_temp,idx=['bins'],columns=['proportion'])
        # data_temp.index.names=[str(index)+"_"+str(cols)+"_env"]
        #save_df(data_temp,workingdir)
        plot_param_evolution(os.path.join(outdir,"food_proportion.pdf"),data_temp,y,title=title,xlab="Time",ylim=False)

        y="times_shared" #can be separated by food type
        title="Total food share average over simulation"
        graphtype="mean" #total shares per timestep
        data_temp=groupby_env(data,["bins"],[y],graphtype,parallel)
        #save_df(data_temp,workingdir)
        plot_param_evolution(os.path.join(outdir,"env_"+y+"_shared.pdf"),data_temp,y,title=title,xlab="Time",ylim=False)

        y="agents_in_cell"
        title="Population average over simulations"
        graphtype="mean"
        data_temp=groupby_env(data,["bins"],[y],graphtype,parallel)
        #save_df(data_temp,workingdir)
        plot_param_evolution(os.path.join(outdir,"env_"+y+"_agents_mean.pdf"),data_temp,y,title=title,xlab="Time",ylim=False)

        y="agents_in_cell"
        title="Group size average over simulations"
        graphtype="sum"
        data_temp=groupby_env(data,["bins"],[y],graphtype,parallel)
        #save_df(data_temp,workingdir)
        plot_param_evolution(os.path.join(outdir,"env_"+y+"_agents_sum.pdf"),data_temp,y,title=title,xlab="Time",ylim=False)

    #####################
    # read reprod files #
    #####################
    print("Processing reproduction files")
    files=glob.glob(os.path.join(datadir,"stats_reprod_*.csv.bz2"))
    if len(files)==0:
        print("Warning: reprod file list is empty. Aborting.")
    else:
        data=pmap(process_reprod_file,files,parallel) # read all files
        data_agg=[d["timeStep"].value_counts() for d in data]
        data_agg=pd.DataFrame(pd.concat(data_agg,copy=False))
        data_agg=compute_stats(data_agg)
        plot_param_evolution(os.path.join(outdir,"reprod.pdf"),data_agg,"timeStep",title="Reproduction frequency",xlab="Timestep",ylab="Number",ylim=False)

    #####################
    # read forage files #
    #####################
    if args.measures:
        print("Processing foraging files")
        files=glob.glob(os.path.join(datadir,"stats_forage_*.csv.bz2"))
        if len(files)==0:
            print("Warning: reprod file list is empty. Aborting.")
        else:
            data=pmap(process_forage_file,files,parallel) # read all files
        ## compute measure of all simulation
        really_do_it=False
        if really_do_it:
            f=functools.partial(groupby_and_pivot_sum,index="ID",columns="food_type",values=None)
            data_pivoted=pmap(f,data,parallel) # for every agent record the skill value at every age
            measures=pmap(compute_entropy_measures,data_pivoted,parallel) # compute measures for each simulation
            res={}
            for l in measures:
                for k,v in l.items():
                    res.setdefault(k,[]).append(v)  # merge results
            res=pd.DataFrame(res)
            res.to_csv(os.path.join(outdir,"measures.csv"),index=False)
        # group by season
        # compute the measure on foraging attempts
        lim=int(args.stable_after[0]) if isinstance(args.stable_after,list) else int(args.stable_after)
        compute_measure_binned(data,"intention",xlim=[lim,lim+5000])
        data=pmap(lambda df:df.loc[df['success']==1],data,parallel) # select only the successful foraging
        # compute the measure on successful foraging
        compute_measure_binned(data,"effective",xlim=[lim,lim+5000])


    ###############################################################################
    # plots for the paper, compare different simulations so must be run only once #
    ###############################################################################

    ### compare measures of phenotype and genotype
    compare_measures("./results/","load_learn/exec/0_special_pql/50","load_nolearn/exec/0_special_nolearn/50","measures_baseline/exec/2_special_nolearn_gener/50","measures_baseline/exec/0_special_nolearn_spec_full/50",'effective',50)
    compare_measures("./results/","load_learn/exec/0_special_pql/50","load_nolearn/exec/0_special_nolearn/50","measures_baseline/exec/2_special_nolearn_gener/50","measures_baseline/exec/0_special_nolearn_spec_full/50",'intention',50)


    # ### compute specialization of agents, determines how often agents forage a certain food type
    # a,b=compute_specializ('./results/wd_load_phen_5000/exec/0_special_pql/50/')
    # c,d=compute_specializ('./results/wd_load_5000/exec/0_special_nolearn/50/')
    # ### agents initialized with the phenotype are specialized: skill_0 is on average a["skill_0_mean"][0]
    # ### a["value_norm_mean"][0] shows how often agents find food of type 0, if it is <0.5 it means food0 is more scarce. But given that the food quantity is the same, it means that agents are consuming it more
    # ### b["value_norm_mean"][0] shows how often agents forage successfully food of type 0, this should be similar to the skill level a["skill_0_mean"][0]
    # ### agents initialized with the genotype are generalists: skill_0 is on average c["skill_0_mean"][0]
    # ### c["value_norm_mean"][0] shows how often agents find food of type 0, if it is 0.5, which is the creation ratio, it means both food types are consumed equally.
    # ### d["value_norm_mean"][0] shows how often agents forage successfully food of type 0, this should be similar to the skill level c["skill_0_mean"][0]
    # ## plot change in gen between learn and nolearn
    plot_convergence_comparison("./results/immortals/exec/",["1_special_pql","2_special_rql","0_special_drl"],3000)
    # plot_skill_surfaces_comparison("./results/exec/",["1_special_pql","0_special_nolearn"],50)
    compute_expediting_comparison('./results/expediting_effect/exec/','0_special_nolearn/3000/','2_special_pql/3000/','1_special_pql/3000',colors=[0.5,0.2,0.8],season_len=3000)
    plot_skill_final_conf_comparison("./results/exec/",["0_special_nolearn","1_special_pql"],["Reactive","Learning"],50,colors=[0.2,0.8])
    # plot_foraging_freq_hist("./results/",["load_learn/exec/0_special_pql/50","load_nolearn/exec/0_special_nolearn/50"],labs=["Learning","Reactive"])
    # plot_foraging_freq_hist("./results/measures_baseline/",["exec/0_special_nolearn_spec_full/50","exec/1_special_nolearn_spec_half/50","exec/2_special_nolearn_gener/50"],labs=["Specialist","Specialist Mixed","Generalist"])
    plot_foraging_freq_hist("./results/",["load_nolearn/exec/0_special_nolearn/50","measures_baseline/exec/1_special_nolearn_spec_half/50","load_learn/exec/0_special_pql/50","measures_baseline/exec/2_special_nolearn_gener/50"],labs=["Reactive","Specialist","Learning","Generalist"],styles=['solid','dashed','solid','dashed'],colors=[0.2,0.2,0.8,0.8])
    plot_skill_final_conf_comparison_stacked("./results/exec/",["0_special_nolearn","1_special_pql"],["Reactive","Learning"],50,colors=[1.0,1.0])
