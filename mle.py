import sys
import os
sys.path.append(os.environ['HOME']+'/work/mpasview')
from mpasview import *

def format_m_to_km(value, tick_number):
    return value/1000.

def plot_yz(mpasodata, varname=None, ax=None, xfrac=0.5, tidx=-1, levels=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    data = mpasodata.dataset.data_vars[varname].isel(Time=tidx).values
    domain = mpasodata.load_variable_map(varname, itime=tidx)
    xmax = domain.x.max()
    xmin = domain.x.min()
    ymax = domain.y.max()
    ymin = domain.y.min()
    cellarea = mpasodata.mesh.acell
    d_cell = np.sqrt(cellarea.mean())
    npoints = int(np.ceil((ymax-ymin)/d_cell))
    print('Nearest neighbor interpolation to {} points.'.format(npoints))
    xmid = xfrac*(xmax-xmin)+xmin
    yy = np.linspace(ymin, ymax, npoints+1)
    xx = np.ones(yy.size)*xmid
    # select nearest neighbor
    pts = np.array(list(zip(xx, yy)))
    tree = spatial.KDTree(list(zip(domain.x, domain.y)))
    p = tree.query(pts)
    cidx = p[1]
    if levels is not None:
        bounds = np.array(levels)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    else:
        norm = None
    fig = ax.contourf(yy, mpasodata.depth, data[cidx,:].transpose(),
                      levels=levels, norm=norm, **kwargs)
    return fig

def plot_yz_mean(mpasodata, varname=None, ax=None, tidx=-1, levels=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    data = mpasodata.dataset.data_vars[varname].isel(Time=tidx).values
    domain = mpasodata.load_variable_map(varname, itime=tidx)
    ymax = domain.y.max()
    ymin = domain.y.min()
    nc = data.shape[0]
    nz = data.shape[1]
    cellarea = mpasodata.mesh.acell
    d_cell = np.sqrt(cellarea.mean())
    npoints = int(np.ceil((ymax-ymin)/d_cell))
    print('Average over x at {} bins in y.'.format(npoints))
    binbnd = np.linspace(ymin, ymax, npoints+1)
    binwidth = (ymax-ymin)/npoints
    dsum = np.zeros([npoints, nz])
    dwgt = np.zeros(npoints)
    mdata = np.zeros([npoints, nz])
    for i in np.arange(nc):
        idx = int((domain.y[i]-binbnd[0]-1.e-6)/binwidth)
        dsum[idx,:] += data[i,:]*cellarea[i]
        dwgt[idx] += cellarea[i]
    for j in np.arange(nz):
        mdata[:,j] = dsum[:,j]/dwgt
    yy = 0.5*(binbnd[0:-1]+binbnd[1:])
    if levels is not None:
        bounds = np.array(levels)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    else:
        norm = None
    fig = ax.contourf(yy, mpasodata.depth, mdata.transpose(),
                      levels=levels, norm=norm, **kwargs)
    return fig

def plot_overview(mpasdata, var, levels, label, tidx=-1, formater=None, bottomdepth=-100, labelcolor='w', **kwargs):
    fig, axarr = plt.subplots(2,1, sharex='col', sharey='row')
    fig.set_size_inches([4,7])
    axarr[0].set_position([0.18, 0.28, 0.8, 0.6])
    axarr[1].set_position([0.18, 0.1, 0.8, 0.18])
    # xy view
    im1 = mpasdata.load_variable_map(
        varname=var, itime=tidx, idepth=0).plot(
        axis=axarr[0], ptype='contourf', levels=levels,
        colorbar=False, swap_xy=True, **kwargs)
    # yz view
    im2 = plot_yz(mpasdata, var, ax=axarr[1], xfrac=1., tidx=tidx, levels=levels, **kwargs)
    # axis properties
    axarr[0].set_aspect(1)
    axarr[0].xaxis.set_major_formatter(plt.FuncFormatter(format_m_to_km))
    axarr[0].yaxis.set_major_formatter(plt.FuncFormatter(format_m_to_km))
    axarr[0].invert_yaxis()
    axarr[0].set_ylabel('x (km)')
    axarr[1].set_ylim([bottomdepth, 0])
    axarr[1].set_ylabel('Depth (m)')
    axarr[1].set_xlabel('y (km)')
    axarr[0].text(0.05, 0.1, '(a)', transform=axarr[0].transAxes, \
                             fontsize=12, color='k', va='top')
    axarr[1].text(0.05, 0.85, '(b)', transform=axarr[1].transAxes, \
                             fontsize=12, color='k', va='top')
    axarr[1].text(0.05, 0.15, str(mpasdata.time[tidx]), transform=axarr[1].transAxes, \
                             color=labelcolor, va='top')
    
    # colorbar
    cax = plt.axes([0.18, 0.85, 0.8, 0.2])
    cax.set_visible(False)
    cb = plt.colorbar(im1, ax=cax, orientation='horizontal', aspect=25)
    cb.set_label(label)
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=40)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    if formater:
        cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(formater))
    return fig
