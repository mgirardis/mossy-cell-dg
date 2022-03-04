import numpy
import copy
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from modules.misc import *
import quantities

def plot_matrix(A,ax=None,interpolation='none',**plotArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    h = ax.imshow(A,interpolation=interpolation,**plotArgs)
    return h, ax, fig

def plot_neuron_activity(time, neuronV, ind=None, **plotArgs):
    n = neuronV.shape[1] # number of columns in neuronV is
    if ind is None:
        ind = numpy.arange(n)
    fig,axe = plt.subplots(nrows=n,ncols=1,sharex=True,sharey=True,figsize=[8,8])
    bMargin = 0.079
    lMargin = 0.1
    w = 1-lMargin-0.01
    h = (1-bMargin) / float(n)
    for i,ax in enumerate(axe):
        ax.plot(time,neuronV[:,i],linewidth=0.2)
        ax.set_ylabel('Mossy ' + str(ind[i]))
        ax.set_position([lMargin,(bMargin + (n-i-1)*h),w,h])
    axe[0].set_xlim(time[0],time[-1])
    return axe,fig

def raster_plot_img(data, simulation=None, ax=None, spk_threshold=59.0, **plotArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    if simulation is None:
        dt = 1
    else:
        dt = simulation['dt']
    cMap = matplotlib.colors.ListedColormap([[1,1,1,1],[0,0,0,1]])
    h = plt.imshow(numpy.transpose(numpy.asarray(data>spk_threshold,dtype=int)),interpolation='none',cmap=cMap,**plotArgs)
    ax.set_aspect('auto')
    plt.xticks(numpy.asarray(ax.get_xticks())*dt)
    ax.set_xlim(0,data.shape[0]*dt)
    return h, ax, fig

def raster_plot(data, simulation=None, ax=None, spk_threshold=59.0, **plotArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    if simulation is None:
        dt = 1
    else:
        dt = simulation['dt']
    t,ev = numpy.nonzero(data>spk_threshold)
    h = plt.plot(t*dt,ev,color=(0.3,0.3,0.3,1),marker='.',markersize=0.5,linestyle='None',**plotArgs)
    return h, ax, fig

#[ax,rh] = plot_space(ax,space,plotArgs)
def plot_space(space, ax=None, **plotArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    if space['type'] == 'Square':
        a = plt.Polygon([[0,0],[0,space['height']],[space['width'],space['height']],[space['width'],0]],closed=True,facecolor=(0.9,0.9,0.9,1),linewidth=2,edgecolor=(0,0,0,1),**plotArgs)
        w = space['width']
        h = space['height']
    else:
        a = plt.Circle((space['radius'],space['radius']),space['radius'],facecolor=(0.9,0.9,0.9,1),linewidth=2,edgecolor=(0,0,0,1),**plotArgs)
        w = 2.0 * space['radius']
        h = w
    ax.add_artist(a)
    ax.set_aspect('equal')
    ax.set_xlim(0,w)
    ax.set_ylim(0,h)
    return ax, fig, a

#[ax,sh,fh,cmap] = plot_rfields(ax,space,rfield,simulation,spaceArgs,rfieldArgs)
def plot_rfields(space, rfield, simulation, cmap_name='viridis', ax = None, spaceArgs = {}, **rfieldArgs):
    ax,fig,sh = plot_space(space,ax=ax,**spaceArgs)
    cmap = get_N_colors(cmap_name, int(simulation['N_cells']))
    #N_rec = len(rfield)
    fh = []
    if not (type(rfield) is list):
        rfield = [rfield]
    for rf in rfield: #for i,rf in enumerate(rfield):
        fh.append(plt.Circle((rf['x0'],rf['y0']),rf['radius'],facecolor=(1,1,1,0),edgecolor=cmap[int(rf['postElement'])]),**rfieldArgs)
        ax.add_artist(fh[-1])
    return ax, fig, sh, fh, cmap

# [th,ah] = plot_trajectory(ax,traj,simulation,plotArgs)
def plot_trajectory(traj,simulation,ax=None,**plotArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    th = []
    ah = []
    n_traj = len(traj)
    cmap = numpy.concatenate((  numpy.reshape(numpy.linspace(0.5,1,n_traj),(n_traj,1)), numpy.zeros(shape=(n_traj,2)), numpy.ones(shape=(n_traj,1))  ),  axis=1)
    T = simulation['totalTime']
    if not (type(traj) is list):
        traj = [traj]
    for i, tr in enumerate(traj):
        x0 = tr['x0']
        y0 = tr['y0']
        vx = tr['vx']
        vy = tr['vy']
        x = [ x0, x0 + vx * T ]
        y = [ y0, y0 + vy * T ]
        dx = 0.001*numpy.diff(x)[0]
        dy = 0.001*numpy.diff(y)[0]
        th.append(plt.plot(x,y,color=cmap[i],linewidth=2,label=('Mouse %d' % i ), **plotArgs ))
        ah.append(plt.arrow( numpy.mean(x)-dx/2.0, numpy.mean(y)-dy/2.0, dx, dy, color=cmap[i], head_width=0.05*numpy.diff(x)[0] ))
    return th, ah

def get_red_gradient(N=None,f=0.0,return_cmap=False):
    c0 = numpy.array((234,183,184,255))/255 
    c1 = numpy.array((237.0,30.0,35.0,255.0))/255
    c2 = numpy.array((114,6,9,255))/255
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('myred',numpy.row_stack((c0,c1,c2)))
    N = 256 if N is None else N
    if return_cmap:
        if f != 0:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('myred',cmap(numpy.linspace(0.0,1.0,int(numpy.ceil(N*(1.0+f)))))[-N:,:])
        return cmap
    if N == 1:
        return c1
    return cmap(numpy.linspace(0.0,1.0,int(numpy.ceil(N*(1.0+f)))))[-N:,:]

def get_gray_gradient(N=None,f=0.0,return_cmap=False):
    c0 = numpy.array((235,235,235,255))/255 
    c1 = numpy.array((80,80,80,255))/255
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mygray',numpy.row_stack((c0,c1)))
    N = 256 if N is None else N
    if return_cmap:
        if f != 0:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('myred',cmap(numpy.linspace(0.0,1.0,int(numpy.ceil(N*(1.0+f)))))[-N:,:])
        return cmap
    if N == 1:
        return cmap(0.5)
    return cmap(numpy.linspace(0.0,1.0,int(numpy.ceil(N*(1.0+f)))))[-N:,:]

def get_hot_colors(N=None,f=0.50):
    """ returns N hot colors (cmap='PuRd'), and ignore the initial f fraction of colors """
    cmap = plt.cm.get_cmap('PuRd')
    if N is None:
        N = 256
        return matplotlib.colors.LinearSegmentedColormap.from_list('myhot',numpy.flipud(cmap(numpy.linspace(0.0,1.0,int(numpy.ceil(N*(1.0+f)))))[-N:,:]))
    else:
        return cmap(numpy.linspace(0.0,1.0,int(numpy.ceil(N*(1.0+f)))))[-N:,:]

def get_cold_colors(N=None,f=0.98):
    """ returns N hot colors (cmap='PuRd'), and ignore the initial f fraction of colors """
    cmap = plt.cm.get_cmap('GnBu')
    if N is None:
        N = 256
        return matplotlib.colors.LinearSegmentedColormap.from_list('myhot',numpy.flipud(cmap(numpy.linspace(0.0,1.0,int(numpy.ceil(N*(1.0+f)))))[-N:,:]))
    else:
        return cmap(numpy.linspace(0.0,1.0,int(numpy.ceil(N*(1.0+f)))))[-N:,:]

def get_N_colors(cmap,N=10):
    if type(cmap) is str:
        cmapr = plt.cm.get_cmap(cmap)
    else:
        cmapr = cmap
    return cmapr(numpy.linspace(0,1,N))

def errorfill(x, y, yerr, fmt='o', color=None, fill_line_color=None, color_fill=None, alpha_fill=0.3, ax=None, label=None, xlabel=None, ylabel=None,**plotArgs):
    if ax is None:
        f = plt.figure()
        ax = f.gca()
    if type(x) is list:
        x = numpy.asarray(x)
    if type(y) is list:
        y = numpy.asarray(y)
    if type(yerr) is list:
        yerr = numpy.asarray(yerr)
    yErrIsNone = type(yerr) is type(None)
    if yErrIsNone:
        ymin,ymax=numpy.nan,numpy.nan
    else:
        if numpy.isscalar(yerr) or (len(yerr) == len(y)):
            ymin = y - yerr
            ymax = y + yerr
        elif len(yerr) == 2:
            ymin, ymax = yerr
    n = y.shape[1] if y.ndim > 1 else 1
    if type(color) is type(lambda:1):
        color = color(n)
    lw = 0 if fill_line_color is None else 1
    default_colors=get_default_colors(n)
    hLine = []
    hFill = []
    if n > 1:
        if (type(label) is list) and (len(label) < n):
            label = repeat_to_complete(label,n)
    for i in range(n):
        xx = x[:,i] if x.ndim > 1 else x
        yy = y[:,i] if y.ndim > 1 else y
        if not yErrIsNone:
            y1 = ymax[:,i] if ymax.ndim > 1 else ymax
            y2 = ymin[:,i] if ymin.ndim > 1 else ymin
            if y1.size < yy.size:
                y1 = extend_vec(y1,yy.size)
            if y2.size < yy.size:
                y2 = extend_vec(y2,yy.size)
        cc = default_colors[i] if color is None else get_color(color,i)
        if n > 1:
            ll = ('%s %d'%(label,i)) if type(label) is str else (label if label is None else label[i])
        else:
            ll = label
        hLine.append(ax.plot(xx, yy, fmt, color=cc, label=ll, **plotArgs))
        if not yErrIsNone:
            cf = hLine[i][0].get_color() if color_fill is None else color_fill
            k1 = numpy.isnan(y1)
            k2 = numpy.isnan(y2)
            y1[k1] = yy[k1]#0.0
            y2[k2] = yy[k2]#0.0
            hFill.append(ax.fill_between(xx, y1, y2, color=cf, alpha=alpha_fill, edgecolor=fill_line_color, linewidth=lw))
    if xlabel or ylabel:
        if (xlabel and ('$' in xlabel)) or (ylabel and ('$' in ylabel)):
            plt.matplotlib.rc('text',usetex=True)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if (xlabel and ('$' in xlabel)) or (ylabel and ('$' in ylabel)):
            plt.matplotlib.rc('text',usetex=False)
    if not yErrIsNone:
        if n == 1:
            hFill = hFill[0]
    return hLine,hFill

def plot_noise_distribution(V,nbins=10,ax=None,**plotArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    P,bins,Vm,sd,Pmax,Pstd = calc_noise_distribution(V,nbins_or_edges=nbins)
    height_at_sigma = numpy.exp(-0.5) / (sd * numpy.sqrt(2.0 * numpy.pi))
        
    h = []
    if type(V) is list:
        db = numpy.abs((bins[1]-bins[0])/2.0)
        h.append(errorfill(bins+db, P, Pstd, fmt='-',ax=ax))
    else:
        h.append(plt.step(bins,P,where='post',label='P(V)',**plotArgs))
    h.append(plt.vlines(Vm,0.0,Pmax,color='k',linewidth=1,linestyle='--'))
    h.append(plt.hlines(height_at_sigma,Vm-sd,Vm+sd,color='k',linewidth=1,linestyle='--'))
    plt.matplotlib.rc('text',usetex=True)
    h.append(plt.text(Vm,Pmax,'$\\bar{V}=%.2f$' % Vm,fontsize=12))
    h.append(plt.text(Vm-sd,height_at_sigma,'$\\bar{V}-\\sigma$',fontsize=12,horizontalalignment='right'))
    h.append(plt.text(Vm+sd,height_at_sigma,'$\\bar{V}+\\sigma$',fontsize=12))
    ax.set_title('$\\bar{V}=%.2f$, $\\sigma=%.2f$' % (Vm,sd),fontsize=12)
    plt.matplotlib.rc('text',usetex=False)
    return P,bins,ax,h

def plot_vertical_lines(x,ax=None,yMin=None,yMax=None,**plotArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    aymin,aymax = ax.get_ylim()
    yMin = aymin if yMin is None else yMin
    yMax = aymax if yMax is None else yMax
    if (type(x) is list) or (type(x) is numpy.ndarray):
        h = []
        for xx in x:
            h.append(plot_vertical_lines(xx, ax=ax, yMin=yMin, yMax=yMax, **plotArgs))
        return h
    else:
        return ax.vlines(x,yMin,yMax,**plotArgs)

def plot_horizontal_lines(y,ax=None,xMin=None,xMax=None,**plotArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    axmin,axmax = ax.get_xlim()
    xMin = axmin if xMin is None else xMin
    xMax = axmax if xMax is None else xMax
    if (type(y) is list) or (type(y) is numpy.ndarray):
        h = []
        for yy in y:
            h.append(plot_horizontal_lines(yy, ax=ax, xMin=xMin, xMax=xMax, **plotArgs))
        return h
    else:
        return ax.hlines(y,xMin,xMax,**plotArgs)

def label_point(x,y,label,ax=None,plotArgs=None,**textArgs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    x_is_list = (type(x) is list) or (type(x) is numpy.ndarray)
    y_is_list = (type(y) is list) or (type(y) is numpy.ndarray)
    if x_is_list and y_is_list:
        if len(x) != len(y):
            raise ValueError('shape of x must equal the shape of y')
    if x_is_list and not y_is_list:
        y = numpy.ones(len(x)) * y
    if y_is_list and not x_is_list:
        x = numpy.ones(len(y)) * x
    lab = label
    th = []
    ph = []
    if x_is_list or y_is_list:
        if not(type(label) is list):
            lab = [label] * len(x)
    for i in range(len(x)):
        th.append(ax.text(x[i],y[i],lab[i],**textArgs))
        if plotArgs:
            ph.append(ax.plot(x[i],y[i],'o',**plotArgs))
    return th,ph

def adjust_lightness(color, factor=0.5):
    import matplotlib.colors as mc
    import colorsys
    if type(color) is list:
        n = len(color)
        c = get_empty_list(n)
        for i in range(n):
            c[i] = adjust_lightness(color[i],factor=factor)
        return c
    else:
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, factor * c[1])), c[2])

def get_color_range_from_lightness(color,factor,n):
    """for each color in the list clist, returns a list of colors beginning at clist element lightened by the factor, until the clist element"""
    if type(color) is list:
        m = len(color)
        c = get_empty_list(m)
        for i in range(m):
            c[i] = get_color_range_from_lightness(color[i],factor=factor,n=n)
        c = numpy.transpose(numpy.asarray(c),(1,0,2))
        return c # c[0] is the original list of colors, c[1] is the list of colors lightened by a small factor, so on and so forth
    else:
        fvec = numpy.linspace(1.0,factor,n)
        nc = get_empty_list(n)
        nc = [ adjust_lightness(color,factor=f) for f in fvec ]
        return nc

def fix_plot_var(x,y,yErr,labels,color_list):
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    yErr = copy.deepcopy(yErr)
    labels = copy.deepcopy(labels)
    color_list = copy.deepcopy(color_list)
    if type(y) is list:
        n = len(y)
        color_list = color_list if type(color_list) is list else repeat_to_complete([color_list],n)
        labels = labels if type(labels) is list else repeat_to_complete([labels],n)
        x = x if type(x) is list else repeat_to_complete([x],n)
        yErr = yErr if type(yErr) is list else repeat_to_complete([yErr],n)
    else:
        n = 1
        color_list = [color_list]
        labels = [labels]
        x = [x]
        y = [y]
        yErr = [yErr]
    return n,x,y,yErr,labels,color_list

def plot_many_panels(x,y,yErr,plotArgs=None,figArgs=None,labels=None,color_list=None,title_txt=None,xlabel_txt=None,ylabel_txt=None,xLim=None,yLim=None,showLegend=None,legendArgs=None):
    if not (type(y) is list):
        raise ValueError('y must be a list, such that each item of y will be plotted in a panel')
    n = len(y)
    figArgs = figArgs if type(figArgs) is dict else dict(nrows=n,ncols=1)
    plotArgs = plotArgs if is_list_of_dict(plotArgs) else ( repeat_to_complete([plotArgs],n) if type(plotArgs) is dict else repeat_to_complete([{}],n) )
    legendArgs = legendArgs if is_list_of_dict(legendArgs) else ( repeat_to_complete([legendArgs],n) if type(legendArgs) is dict else repeat_to_complete([{}],n) )
    xLim = xLim if is_list_of_list(xLim) else ( fill_to_complete([xLim],None,n) if type(xLim) is list else get_empty_list(n) )
    yLim = yLim if is_list_of_list(yLim) else ( fill_to_complete([yLim],None,n) if type(yLim) is list else get_empty_list(n) )
    fig,ax = plt.subplots(**figArgs)
    if type(ax) is numpy.ndarray:
        ax = ax.T.flatten()
    title_txt = fill_to_complete(title_txt,None,n)
    xlabel_txt = fill_to_complete(xlabel_txt,None,n)
    ylabel_txt = fill_to_complete(ylabel_txt,None,n)
    showLegend = fill_to_complete(showLegend,None,n)
    color_list = fill_to_complete(color_list,None,n)
    xLim = fill_to_complete(xLim,None,n)
    yLim = fill_to_complete(yLim,None,n)
    for i in range(n):
        ncurves,xx,yy,yyErr,labels_,color_list_ = fix_plot_var(x[i],y[i],yErr[i],labels[i],color_list[i])
        for j in range(ncurves):
            errorfill(xx[j],yy[j],yyErr[j],color=color_list_[j],label=labels_[j],ax=ax[i],**plotArgs[i])
    plt.matplotlib.rc('text',usetex=True)
    for i in range(n):
        if title_txt[i]:
            ax[i].set_title(title_txt[i])
        if xlabel_txt[i]:
            ax[i].set_xlabel(xlabel_txt[i])
        if ylabel_txt[i]:
            ax[i].set_ylabel(ylabel_txt[i])
        if xLim[i]:
            ax[i].set_xlim(*xLim[i])
        if yLim[i]:
            ax[i].set_ylim(*yLim[i])
        if showLegend[i]:
            ax[i].legend(**legendArgs[i])
    plt.matplotlib.rc('text',usetex=False)
    return ax

def plot_two_panel_comparison(xtop,ytop,yErrtop,xbot,ybot,yErrbot,topPlotArgs=None,bottomPlotArgs=None,figArgs=None,top_labels=None,bottom_labels=None,top_color_list=None,bottom_color_list=None,title_txt=[None,None],xlabel_txt=[None,None],ylabel_txt=[None,None],showLegend=[1,1],xLim=[None,None],yLim=[None,None],legendArgs=[None,None]):
    figArgs = dict(nrows=2,ncols=1,**figArgs) if type(figArgs) is dict else dict(nrows=2,ncols=1)
    topPlotArgs = topPlotArgs if type(topPlotArgs) is dict else {}
    bottomPlotArgs = bottomPlotArgs if type(bottomPlotArgs) is dict else {}
    return plot_many_panels([xtop,xbot],[ytop,ybot],[yErrtop,yErrbot],plotArgs=[topPlotArgs,bottomPlotArgs],figArgs=figArgs,labels=[top_labels,bottom_labels],
        color_list=[top_color_list,bottom_color_list],title_txt=title_txt,xlabel_txt=xlabel_txt,ylabel_txt=ylabel_txt,showLegend=showLegend,xLim=xLim,yLim=yLim,legendArgs=legendArgs)
    # fig,ax = plt.subplots(**figArgs)
    # ntop,x_top,y_top,yErr_top,top_labels_,top_color_list_ = fix_plot_var(xtop,ytop,yErrtop,top_labels,top_color_list)
    # nbot,x_bot,y_bot,yErr_bot,bottom_labels_,bottom_color_list_ = fix_plot_var(xbot,ybot,yErrbot,bottom_labels,bottom_color_list)
    # title_txt = fill_to_complete(title_txt,None,2)
    # xlabel_txt = fill_to_complete(xlabel_txt,None,2)
    # ylabel_txt = fill_to_complete(ylabel_txt,None,2)
    # for i in range(ntop):
    #     errorfill(x_top[i],y_top[i],yErr_top[i],color=top_color_list_[i],label=top_labels_[i],ax=ax[0],**topPlotArgs)
    # for i in range(nbot):
    #     errorfill(x_bot[i],y_bot[i],yErr_bot[i],color=bottom_color_list_[i],label=bottom_labels_[i],ax=ax[1],**bottomPlotArgs)
    # plt.matplotlib.rc('text',usetex=True)
    # for i in range(2):
    #     if title_txt[i]:
    #         ax[i].set_title(title_txt[i])
    #     if xlabel_txt[i]:
    #         ax[i].set_xlabel(xlabel_txt[i])
    #     if ylabel_txt[i]:
    #         ax[i].set_ylabel(ylabel_txt[i])
    # plt.matplotlib.rc('text',usetex=False)
    # ax[0].legend()
    # ax[1].legend()
    # return ax

def plot_curves_with_colormap(x,y,paramVal,yErr=None,cmap_name='jet',xlabel_txt='',ylabel_txt='',clabel_txt='',title_txt='',clabel_fontsize=10,ax=None,**plotArgs):
    # each column in y corresponds to each element in paramVal
    figsize = numpy.asarray([8.0,4.0])
    cax_pos = numpy.asarray([0.64,0.70,0.01,0.15])
    ax_pos = numpy.asarray([0.1,0.11,0.6,0.8])
    if ax:
        fig = ax.figure
        s = fig.get_size_inches()
        p = ax.get_position()
        T = numpy.zeros((2,2))
        T[0,0] = p.width / ax_pos[2]
        T[1,1] = p.height / ax_pos[3]
        Tcax = numpy.matmul(T,(cax_pos[0:2]-ax_pos[0:2]))
        #cax_pos[0] = p.x0 + (cax_pos[0]-ax_pos[0])*r_fig[0]
        #cax_pos[1] = p.y0 + (cax_pos[1]-ax_pos[1])*r_fig[1]
        r_fig =  figsize / s
        cax_pos[2] = cax_pos[2]*r_fig[0]
        cax_pos[3] = cax_pos[3]*r_fig[1]
        cax_pos[0] = p.x0 + Tcax[0] - cax_pos[2]
        cax_pos[1] = p.y0 + Tcax[1] - cax_pos[3]
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.set_position(ax_pos)
    nCurves = 1 if is_vector(y) else y.shape[1]
    colors = get_N_colors(cmap_name,N=nCurves) # number of curves is the number of columns in y
    errorfill(x,y,yErr,ax=ax,color=colors,**plotArgs)
    cax = fig.add_axes(cax_pos,label='color_'+get_hash_str(x,y,paramVal,yErr,cmap_name,xlabel_txt,ylabel_txt,clabel_txt,title_txt,ax,plotArgs))
    if numpy.isscalar(paramVal):
        paramVal = numpy.asarray( [ paramVal.copy(), paramVal.copy()+1.0 ] )
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(paramVal[0],paramVal[-1]), cmap=cmap_name), ax=ax, cax=cax)
    if clabel_txt:
        cax.set_title(clabel_txt,fontsize=clabel_fontsize)
    if title_txt:
        ax.set_title(title_txt)
    plt.matplotlib.rc('text',usetex=True)
    if xlabel_txt:
        ax.set_xlabel(xlabel_txt)
    if ylabel_txt:
        ax.set_ylabel(ylabel_txt)
    plt.matplotlib.rc('text',usetex=False)
    if nCurves == 1:
        del cax
        cax = None
    return ax,cax

def plot_complete_voltage_trace(trial_idx=0,simArgs=None,plotTitle=None,color_list=None,ax=None,legend=None,maxfactor=0.1,**neuronVars):
    color_list = get_default_colors() if color_list is None else color_list
    neuronType = simArgs['neuronType'] if simArgs else None
    t = neuronVars['t']
    V = neuronVars['V']
    g1 = neuronVars['g1']
    g2 = neuronVars['g2']
    th = neuronVars['th']
    I = neuronVars['I']
    if (type(V) is list) and type(V[0]) is list:
        n = len(V)
        ax = None
        keys = neuronVars.keys()
        new_color_list = get_color_range_from_lightness(color_list,maxfactor,n)
        if not legend:
            legend = ['$I_0={:g}$'.format(numpy.max(I[i])) for i in range(n)]
        for i in range(n):
            nvars = dict(zip(keys, [ (neuronVars[k][i] if type(neuronVars[k]) is list else neuronVars[k]) for k in keys ] ))
            ax=plot_complete_voltage_trace(trial_idx=trial_idx,simArgs=simArgs,plotTitle=plotTitle,color_list=new_color_list[i],ax=ax,legend=legend,**nvars)
        return ax
    if ax is None:
        fig,ax = plt.subplots(nrows=5,ncols=1,sharex=True,sharey=False,figsize=[8,7])
    ax[0].plot(t,V[trial_idx],linewidth=0.2,color=color_list[0])
    ax[0].set_position([0.1,0.64,0.89,0.32])
    ax[1].plot(t,g2[trial_idx],linewidth=1,color=color_list[4])
    ax[1].set_position([0.1,0.52,0.89,0.1])
    ax[2].plot(t,g1[trial_idx],linewidth=1,color=color_list[3])
    ax[2].set_position([0.1,0.42,0.89,0.1])
    ax[3].plot(t,th[trial_idx],linewidth=1,color=color_list[2])
    ax[3].set_position([0.1,0.20,0.89,0.2])
    ax[4].plot(t,I,linewidth=1,color=color_list[1])
    ax[4].set_position([0.1,0.08,0.89,0.1])
    ax[0].set_xlim(t[0],t[-1])
    ax[0].set_ylabel((neuronType + ', ' if neuronType else '') + 'V(t), mV')
    ax[3].set_ylabel('Threshold, mV', fontsize=8)
    ax[4].set_ylabel('Current, nA', fontsize=8)
    ax[4].set_xlabel('Time (ms)', fontsize=8)
    plt.matplotlib.rc('text',usetex=True)
    ax[1].set_ylabel('$g_2$, nA', fontsize=8)
    ax[2].set_ylabel('$g_1$, nA', fontsize=8)
    if legend:
        ax[0].legend(legend)
    if plotTitle:
        ax[0].set_title(plotTitle,fontsize=10)
    plt.matplotlib.rc('text',usetex=False)
    return ax

def plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,simParam,same_panel_V_theta=False,show_tspk_curve=True,figsize=(6,6)):
    I[I==0] = numpy.nan
    DeltaTexp = numpy.resize(DeltaTexp,len(DeltaTexp)+1)
    DeltaTexp[-1] = DeltaTexp[-1] + DeltaTexp[-2]
    fig,ax = plt.subplots(nrows=4,ncols=1,figsize=figsize)
    ax[0].plot(t,V,'-b',linewidth=0.8)
    plot_vertical_lines(tOff+DeltaTexp,ax=ax[0],linewidth=0.5,linestyle='--',color='k')
    plot_vertical_lines(tOff,ax=ax[0],linewidth=1.5,linestyle='--',color='#C55A11')
    if same_panel_V_theta:
        ax[0].plot(t,theta,'-',c='tab:green',linewidth=0.8)
    else:
        ax[1].plot(t,theta,'-',c='tab:green',linewidth=0.8)
        plot_vertical_lines(tOff+DeltaTexp,ax=ax[1],linewidth=0.5,linestyle='--',color='k')
        plot_vertical_lines(tOff,ax=ax[1],linewidth=1.5,linestyle='--',color='#C55A11')
    ax[2].plot(t,I,'-r',linewidth=0.8)
    plot_vertical_lines(tOff+DeltaTexp,ax=ax[2],linewidth=0.5,linestyle='--',color='k')
    plot_vertical_lines(tOff,ax=ax[2],linewidth=1.5,linestyle='--',color='#C55A11')
    if show_tspk_curve:
        errorfill(DeltaT, th_amp, th_std, fmt=':o',ax=ax[3],label='$\\theta$ vs. $t_{spk}$')
    errorfill(DeltaTexp[:-1], th_amp, th_std, fmt=':s',color='k',ax=ax[3],label='$\\theta$ vs. $D$') # '#C55A11'
    if same_panel_V_theta:
        ax[0].set_position([0.1,0.60,0.89,0.37])
        ax[1].set_axis_off()
        ax[1].set_visible(False)
    else:
        ax[0].set_position([0.1,0.76,0.89,0.21])
        ax[1].set_position([0.1,0.60,0.89,0.16])
        ax[1].set_ylabel('$\\theta(t)$')
        ax[1].set_xlim(*ax[0].get_xlim())
        ax[1].sharex(ax[0])
    ax[2].set_position([0.1,0.47,0.89,0.13])
    ax[3].set_position([0.1,0.08,0.89,0.31])
    plt.matplotlib.rc('text',usetex=True)
    ax[0].set_ylabel(simParam['neuronType'] + ', $V(t)$')
    ax[2].set_ylabel('$I(t)$')
    ax[3].set_ylabel('$\\theta-\\theta_0$')
    ax[3].set_xlabel('Delay $D$ or $t_{spk}$ (ms)')
    ax[3].legend()
    plt.matplotlib.rc('text',usetex=False)

    xt = ax[2].get_xticks()

    xt_new = xt[numpy.logical_and(xt>=0,xt <= tOff)]
    n = len(xt_new)
    xt_new = numpy.resize(xt_new,n + len(DeltaTexp))
    xt_new[n:] = tOff + DeltaTexp
    ax[2].set_xticks(xt_new)
    xtl = ax[2].get_xticks().tolist()
    D0 = DeltaTexp[1] - DeltaTexp[0]
    xtl[n:] = tOff + Iprobe_duration + DeltaTexp - D0
    ax[2].set_xticklabels([int(l) for l in xtl ])
    ax[2].set_xlabel('Time (ms)')
    ax[2].set_xlim(*ax[0].get_xlim())
    ax[2].sharex(ax[0])
    return ax

def plot_param_search_var(x_exp,y_exp,yErr_exp,curr_exp,modelDataStruct,cLabelSpk_exp='Exp. $I_{inj}$ (pA)',cLabelSpk_mod='Model $I_{inj}$ (pA)',cLabelCurr_exp='Exp. Spk #',cLabelCurr_mod='Model Spk #',xlabelSpk_txt='Spike #',xlabelCurr_txt='Input current (pA)',ylabel_txt='',xLim_spk=None,xLim_curr=None,fmtSpk_exp=':s',fmtSpk_mod='--o',fmtCurr_exp=':s',fmtCurr_mod='--o',maxSpk='experiment',rescale_currents=False):
    """
    experiments input: measured in pA
    model input current: measured in nA (converted to pA for plotting)
    """
    if not (is_vector(y_exp) or is_vector(modelDataStruct['avg'])):
        fig,ax = plt.subplots(nrows=2,ncols=1,sharex=False,sharey=False,figsize=[6,6])
        plt.tight_layout()
    else:
        fig,ax1 = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False)
        ax = numpy.asarray([ ax1 ])
    def plot_data(ax,x,y,yErr,parVal,linestyle,data_label,colorLabel,color_func):
        if is_vector(y):
            errorfill(x,y,yErr,fmt=linestyle, label=data_label.format(parVal),color=color_func(N=1),markersize=3.3,ax=ax,alpha_fill=0.1)
            cax=None
        else:
            err_color = color_func(N=50,f=0.25)[-1,:].flatten()
            _,cax = plot_curves_with_colormap(x,y,parVal,yErr=yErr,cmap_name=color_func(), \
                                    clabel_txt=colorLabel, clabel_fontsize=10, fmt=linestyle,markersize=3.3,color_fill=err_color,alpha_fill=0.1,ax=ax)
        return cax
    cax10 = plot_data(ax[0],x_exp,y_exp,yErr_exp,curr_exp,fmtSpk_exp,'Exp. $\\bar{{I}}={:g}$ pA',cLabelSpk_exp,get_hot_colors) # plotting vs. spike for each current
    cax20 = plot_data(ax[0],modelDataStruct['spk'],modelDataStruct['avg'],modelDataStruct['std'],modelDataStruct['I']*1.0e3,fmtSpk_mod,'Model $\\bar{{I}}={:g}$ pA',cLabelSpk_mod,get_cold_colors)
    if len(ax) > 1:
        maxSpk = modelDataStruct['spk'][-1] if maxSpk is None else (int(x_exp[-1]) if str(maxSpk).lower() == 'experiment' else maxSpk)
        new_current_range = [0.0,1.0] if rescale_currents else None
        Idata_exp = linearTransf(curr_exp,new_current_range) if rescale_currents else curr_exp
        Idata_model = linearTransf(modelDataStruct['I']*1.0e3,new_current_range) if rescale_currents else modelDataStruct['I']*1.0e3
        cax11 = plot_data(ax[1],Idata_exp,y_exp.T,yErr_exp.T,x_exp,fmtCurr_exp,'Exp. $\\overline{{Spk}}={:g}$',cLabelCurr_exp,get_hot_colors) # plotting vs. spike for each current
        cax21 = plot_data(ax[1],Idata_model,modelDataStruct['avg'].T[:,:maxSpk],modelDataStruct['std'].T[:,:maxSpk],modelDataStruct['spk'][:maxSpk],fmtCurr_mod,'Model $\\overline{{spk}}={:g}$',cLabelCurr_mod,get_cold_colors)
    else:
        cax11 = None
        cax21 = None
    plt.matplotlib.rc('text',usetex=True)
    ax[0].set_xlabel(xlabelSpk_txt)
    ax[0].set_ylabel(ylabel_txt)
    if len(ax) > 1:
        ax[1].set_xlabel(xlabelCurr_txt)
        ax[1].set_ylabel(ylabel_txt)
    plt.matplotlib.rc('text',usetex=False)
    if xLim_spk:
        ax[0].set_xlim(*xLim_spk)
    if xLim_curr and (len(ax) > 1):
        ax[1].set_xlim(*xLim_curr)
    if is_vector(y_exp) or is_vector(modelDataStruct['avg']):
        ax[0].legend()
    if cax10 and cax20:
        cax10_pos = cax10.get_position().bounds
        cax20.set_position([cax10_pos[0]-cax10_pos[2]*12,cax10_pos[1],cax10_pos[2],cax10_pos[3]])
    if cax11 and cax21:
        cax11_pos = cax11.get_position().bounds
        cax21.set_position([cax11_pos[0]-cax11_pos[2]*12,cax11_pos[1],cax11_pos[2],cax11_pos[3]])
    plt.tight_layout()
    return ax

# def animate(t,img,plt,L):
#     pdata = plt.imshow(img[t])
#     ax = plt.gca()
#     ax.set_xticks(numpy.arange(L[1]))
#     ax.set_yticks(numpy.arange(L[0]))
#     #print('t = %d'%t)
#     #print(img[t])
#     return pdata,

def get_color(clist,k):
    if type(clist) is list:
        return clist[k%len(clist)]
    elif type(clist) is numpy.ndarray:
        if len(clist.shape) == 1:
            return clist
        else:
            return clist[k%clist.shape[0],:]
    return clist

def get_default_colors(N=None):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    clist = []
    for c in colors:
        clist.append(c)
    if N is None:
        return clist
    else:
        if N > len(clist):
            return repeat_to_complete(clist,N)
        else:
            return clist[:N]
