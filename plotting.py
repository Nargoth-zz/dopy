import os.path
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from . import statistics
from . import selection

def fill_between_steps(ax, x, y1, y2=0, step_where='pre', **kwargs):
    ''' fill between a step plot and

    Parameters
    ----------
    ax : Axes
       The axes to draw to

    x : array-like
        Array/vector of index values.

    y1 : array-like or float
        Array/vector of values to be filled under.
    y2 : array-Like or float, optional
        Array/vector or bottom values for filled area. Default is 0.

    step_where : {'pre', 'post', 'mid'}
        where the step happens, same meanings as for `step`

    **kwargs will be passed to the matplotlib fill_between() function.

    Returns
    -------
    ret : PolyCollection
       The added artist

    '''
    if step_where not in {'pre', 'post', 'mid'}:
        raise ValueError("where must be one of {{'pre', 'post', 'mid'}} "
                         "You passed in {wh}".format(wh=step_where))

    # make sure y values are up-converted to arrays
    if np.isscalar(y1):
        y1 = np.ones_like(x) * y1

    if np.isscalar(y2):
        y2 = np.ones_like(x) * y2

    # temporary array for up-converting the values to step corners
    # 3 x 2N - 1 array

    vertices = np.vstack((x, y1, y2))

    # this logic is lifted from lines.py
    # this should probably be centralized someplace
    if step_where == 'pre':
        steps = ma.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, 0::2], steps[0, 1::2] = vertices[0, :], vertices[0, :-1]
        steps[1:, 0::2], steps[1:, 1:-1:2] = vertices[1:, :], vertices[1:, 1:]

    elif step_where == 'post':
        steps = ma.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]

    elif step_where == 'mid':
        steps = ma.zeros((3, 2 * len(x)), np.float)
        steps[0, 1:-1:2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 2::2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 0] = vertices[0, 0]
        steps[0, -1] = vertices[0, -1]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :]
    else:
        raise RuntimeError("should never hit end of if-elif block for validated input")

    # un-pack
    xx, yy1, yy2 = steps

    # now to the plotting part:
    return ax.fill_between(xx, yy1, y2=yy2, **kwargs)

def plot_steps_with_errors(binning, y_bins, errors, color = None, label = None, alpha = 0.25):
  bin_centers = 0.5*(binning[1:] + binning[:-1])
  bin_width = binning[2] - binning[1]
  barplot = plt.errorbar(bin_centers,
                         y_bins,
                         yerr = errors,
                         linestyle = 'steps-mid',
                         marker = '.',
                         label = label,
                         color = color)

  if alpha != 0:
    fill_between_steps(plt.gca(), bin_centers, y_bins, step_where='mid',
                       alpha=alpha,
                       color=barplot[0].get_color())


def plot_with_selection(plotname, component_to_observable, selection, component_order, selectionorder,
                        component_infos, print_efficiencies=False, print_single_cut_efficiencies=False):
  cutframes  = {}
  histranges = {}
  global_min, global_max = None, None
  used_plotvars = []

  for component_identifier in component_order:
    if component_identifier in component_to_observable:
      fr = component_infos[component_identifier]["dataframe"]
      plotvar = component_to_observable[component_identifier]

      if plotvar not in fr:
        print("ERROR requested to plot {} which is not in {}".format(plotvar, component_identifier))
        print("ERROR continuing with next dataset")
        continue

      if print_efficiencies or print_single_cut_efficiencies:
        print("Plot: {:<15}  Dataset: {:<15}  Plotvar: {:<15}".format(plotname, component_identifier, plotvar))
        print("-----------------------------------------------------------------------------")
      fr_cut = fr
      if selection != {}:
        if type(list(selection.values())[0]) == type({}): #Then we need to get the component first
          if component_identifier in selection:
            sel = selection[component_identifier]
          else:
            print("No selection for component {}".format(component_identifier))
        else: #Apply same selection for all components
          sel = selection
        fr_cut = selection.apply_selection_to_dataframe(fr, sel, selectionorder, print_efficiencies=print_efficiencies,
                                                              print_single_cut_efficiencies=print_single_cut_efficiencies)
      cur_min = fr_cut[plotvar].min()
      cur_max = fr_cut[plotvar].max()
      cutframes[component_identifier] = fr_cut
      global_min = cur_min if ((not global_min) or (cur_min < global_min)) else global_min
      global_max = cur_max if ((not global_max) or (cur_max > global_max)) else global_max

      used_plotvars.append(plotvar)
      print("\n")
    else:
      continue

  for component_identifier in component_order:
    if component_identifier in component_to_observable:
      if component_identifier in cutframes:
        y, bins = np.histogram(cutframes[component_identifier][plotvar].values, bins=100, range=(global_min,global_max))
        errors = statistics.poissonian_cls(y)
        y, errors = statistics.normalize_histogram(y, errors)
        plot_steps_with_errors(bins, y, errors, label=component_identifier, alpha=0.1)
        #plt.tight_layout()
      else:
        print("ERROR requested to plot {} which is not there".format(component_identifier))

  plt.legend(loc='best')
  fig = plt.gcf()
  plt.xlabel(set(used_plotvars))
  plt.title(plotname)
  fig.set_size_inches(8,6)

def plot_with_plot_config_selection(plotconfig, plotname_to_selection, selectionorder,
                                    component_infos, plot_order, component_order, print_efficiencies=False,
                                    print_single_cut_efficiencies=False, allplotsfile=None, savepath=None,
                                    showplots=False):
  """
  Producing a series of plots from a plotconfig object and other information.
  I need to document this properly...

  plotconfig: dict -- str : {str : str}
    maps "plotname" : {"component1" : "obs1", "component2" : "obs2", ...}
  plotname_to_selection: dict - str : {str : [float, float]}
    maps "plotname" : {"obs1" : [min, max], "obs2" : [min, max], ...}
  plot_order: list of str
    defines the order of plots
  selection_order: list of str
    defines the order in which selection is applied
  component_infos: dict -- str : {str : various}
    maps component_identifier to dictionary of properties (str) and their values (various)
  """
  for plotname in plot_order:
    if plotname in plotname_to_selection:
      selection = plotname_to_selection[plotname]
    else:
      selection = {}
    plot_with_selection(plotname, plotconfig[plotname], selection, component_order,
                        selectionorder, component_infos, print_efficiencies, print_single_cut_efficiencies)
    if allplotsfile:
      plt.savefig(allplotsfile, format='pdf')
    if savepath:
      plt.savefig(os.path.join(savepath,plotname+'.pdf'), format='pdf')
    if showplots:
      plt.show()
    plt.clf()
