import collections
import copy
import os
import os.path

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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


class PlotComponent:
    def __init__(self, name, data, observable, mothername="None"):
        self.mother_name      = mothername
        self.name             = name
        self.data             = data
        self.observable       = observable
        self.initial_size     = len(self.data)

        self.x_min            = None    # Is this the way one does this?
        self.x_max            = None
        self.update_x_min_max()

        self.dict_selection   = collections.OrderedDict()
        self.string_selection = ""


    def prepare_selection(self, selection):
        if type(selection) == type({}):
            self.dict_selection.update(selection)
            return self.dict_selection
        elif type(selection) == type(""):
            if self.string_selection == "":
                self.string_selection = selection
                return self.string_selection
            else:
                raise Exception("""String like selection is already set - it is unclear how to combine
                                {} with {} for plot {} component {}""".format(selection, self.string_selection, self.mother_name, self.name))
        else:
            raise Exception("Unknown selection type")


    def apply_selection(self, print_efficiencies=False, print_single_cut_efficiencies=False):
        
        if self.dict_selection != {}:
            selectionorder = list(self.dict_selection.keys())
            self.data = selection.apply_selection_to_dataframe(self.data, self.dict_selection, selectionorder,
                                                                     print_efficiencies=print_efficiencies,
                                                                     print_single_cut_efficiencies=print_single_cut_efficiencies)
        if self.string_selection != "":
            print("WARNING: Using experimental string selection style for {}".format(self.string_selection))
            if(print_efficiencies or print_single_cut_efficiencies):
                print("WARNING: Efficiencies for experimental string selection style are not implemented yet.")
            self.data = self.data.query(self.string_selection)

        if print_efficiencies or print_single_cut_efficiencies:
            print("\n")

        self.update_x_min_max()


    def update_x_min_max(self):
        self.x_min = self.data[self.observable].min()
        self.x_max = self.data[self.observable].max()


    def get_min(self): return self.x_min
    def get_max(self): return self.x_max



class Plot:
    def __init__(self, title):
        self.components        = collections.OrderedDict()
        self.title             = title
        self.observables       = []
        self.x_min             = None
        self.x_max             = None

        self.range_auto_update       = True
        self.range_part_of_selection = False

        self.is_selected       = False
        self.selection_changed = False


    def add_component(self, data, observable, component_name=""):
        if component_name=="":
            component_name = str(len(self.components))
        self.components[component_name] = PlotComponent(component_name, data, observable, mothername=self.title)
        self.observables.append(observable)


    def prepare_selection(self, selection):
        for component_name, component in self.components.items():
            selection_for_component = selection
            if type(selection) == type({}): # Ramon's selection style with dicts
                if selection_for_component != {}:
                    if type(list(selection.values())[0]) == type({}): #Then we need to get the component first
                        if component.name in selection:
                            selection_for_component = selection[component.name]
                        else:
                            selection_for_component = {}
                            print("WARNING: No selection for component {} but nested selection passed.".format(component.name))
                    else: #Apply same selection for all components
                        print("INFO: Will apply same selection to all components")
            component.prepare_selection(selection_for_component)
        # Remember if this plot was selected before and the selection was changed after that
        if self.is_selected:
            print("WARNING: preparing the selection for {} which has already been selected.".format(self.title))
            self.selection_changed = True



    def apply_selection(self, print_efficiencies=False, print_single_cut_efficiencies=False):
        if not self.range_auto_update and not self.range_part_of_selection:
            print("WARNING: Range for plot {} has been manually set with range_part_of_selection=False.".format(self.title))
            print("WARNING: Efficiencies are calculated against the full complete dataset.")
        for component_name, component in self.components.items():
            if print_efficiencies or print_single_cut_efficiencies:
                if component.dict_selection or component.string_selection:
                    if(self.is_selected):
                        print("WARNING {} {} has been selected before. Efficiencies are not normalized to the total data".format(self.title,
                                                                                                                             component_name))
                    print("Selections for Plot: {:<15}  Component: {:<15}".format(self.title, component.name))
                    print("-----------------------------------------------------------------")                
            component.apply_selection(print_efficiencies, print_single_cut_efficiencies)
        self.is_selected = True


    def set_range(self, min, max, add_to_selection=False):
        if add_to_selection:
            for component_name, component in self.components.items():
                component.prepare_selection({component.observable : [min, max]})

        self.range_auto_update = False
        self.x_min = min
        self.x_max = max

        self.range_part_of_selection = add_to_selection



    def print_components(self):
        print("Plot: {}".format(self.title))
        print("{:<10} {:<10} {:<10}".format("component", "rows", "columns"))
        for component in self.components:
            print("{:<10} {:<10} {:<10}".format(component.name, len(component.data),
                                                len(component.data.columns)))


    def plot(self, verbose=False):
        if self.selection_changed or not self.is_selected:
            if verbose:
                self.apply_selection(print_efficiencies=True,
                                     print_single_cut_efficiencies=True)
            else:
                self.apply_selection()
        elif verbose:
            print("WARNING: Plot {} already has selection applied - won't apply again.".format(self.title))
        
        if self.range_auto_update:
            for component_name, component in self.components.items():
                x_min = component.get_min()
                x_max = component.get_max()
                if (not self.x_min) or (x_min < self.x_min):
                    self.x_min = x_min
                if (not self.x_max) or (x_max > self.x_max):
                    self.x_max = x_max

        for component_name, component in self.components.items():
            y, bins = np.histogram(component.data[component.observable].values, bins=100, range=(self.x_min,self.x_max))
            errors = statistics.poissonian_cls(y)
            y, errors = statistics.normalize_histogram(y, errors)
            plot_steps_with_errors(bins, y, errors, label=component.name, alpha=0.1)

        plt.legend(loc='best')
        fig = plt.gcf()
        plt.xlabel(set(self.observables))
        plt.title(self.title)

        fig.set_size_inches(8,6)


    def copy(self):
        plot_copy = Plot(self.title)
        
        plot_copy.components              = copy.deepcopy(self.components)
        plot_copy.observables             = self.observables.copy()
        plot_copy.x_min                   = self.x_min
        plot_copy.x_max                   = self.x_max
        plot_copy.range_auto_update       = self.range_auto_update
        plot_copy.range_part_of_selection = self.range_part_of_selection
        plot_copy.is_selected             = self.is_selected

        return plot_copy



class Plotter:
    def __init__(self, savepath=None):
        self.plots    = collections.OrderedDict()
        self.savepath = savepath

        self.is_finalized = False
        if savepath:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            all_plots_file = os.path.join(savepath, "AllPlots.pdf")
            self.pdfpages_file = PdfPages(all_plots_file)
        else:
            self.pdfpages_file = None


    def __getitem__(self, plot_name):
        return self.plots[plot_name]


    def create_plot(self, plot_name, datasets, observable, component_labels=[]):
        """ Creates single plot of an observables in multiple datasets
        """
        if not type(datasets)==type([]):
            datasets = [datasets]
        
        if component_labels and len(component_labels) != len(datasets):
            raise Exception('length of component_labels does not match number of components for {}'.format(plot_name))

        plot = Plot(plot_name)
        
        if component_labels:
            for dataset, component_label in zip(datasets, component_labels):
                plot.add_component(dataset, observable, component_label)
        else:
            for dataset in datasets:
                plot.add_component(dataset, observable)

        self.plots[plot_name] = plot
        return plot


    def create_plots(self, datasets, observables, plot_names=[], component_labels=[]):
        """ Creates multiple plots of the same observables in multiple datasets
        """
        if not plot_names:
            plot_names = observables
        elif len(plot_names) != len(observables):
            raise Exception("number of datasets doesn't match length of number of plot names")

        created_plots = []
        for plot_name, observable in zip(plot_names, observables):
            plot = self.create_plot(plot_name, datasets, observable, component_labels)
            created_plots.append(plot)

        return created_plots


    def duplicate_plot(self, old_plot_name, new_plot_name, selection=None):
        """ Duplicates plot and applies a selection to the copy
        """
        if old_plot_name in self.plots:
            self.plots[new_plot_name] = self.plots[old_plot_name].copy()
            self.plots[new_plot_name].title = new_plot_name
            if selection:
                self.apply_selection_to_plot(new_plot_name, selection)
            return self.plots[new_plot_name]
        else:
            print("ERROR: couldn't duplicate because {} was not found.".format(old_plot_name))


    def duplicate_plots(self, suffix_to_append="copy", plots=[], selection=None):
        """ Duplicates multiple plots and applies a selection to the copy
        """
        if plots == []:
            plots = list(self.plots.keys())

        duplicated_plots = []
        for old_plot_name in plots:
            new_plot_name  = old_plot_name+ "_" + suffix_to_append
            duplicate_plot = self.duplicate_plot(old_plot_name, new_plot_name, selection)
            duplicated_plots.append(duplicate_plot)

        return duplicated_plots


    def prepare_selection_for_plot(self, plot_name, selection):
        """ Applies selection to a single plot
        """
        if plot_name in self.plots:
            self.plots[plot_name].prepare_selection(selection)
        else:
            print("Plotter::prepare_selection_for_plot couldn't apply selection plot {} not found".format(plot_name))

        return self.plots[plot_name]


    def set_range_for_plot(self, plot_name, min, max, add_to_selection=False):
        self.plots[plot_name].set_range(min, max, add_to_selection)


    def apply_selection_to_plots(self, selection):
        """ Applies selection to all plots
        """
        for plot_name in self.plots:
            self.plots[plot_name].apply_selection()


    def print_plots(self):
        """ Printout of plots with their components
        """
        for plot_name in self.plots:
            plot.print_components()


    def get_plots(self):
        """ Returns all plots that are currently prepared
        """
        return plots


    def plot(self, show_plots=True, verbose=False):
        """ Plot all plots
        """
        if self.is_finalized:
            raise Exception("Plotter already finalized!")

        for plot_name,plot in self.plots.items():
            plot.plot(verbose)
            if self.savepath:
                plt.savefig(self.pdfpages_file, format='pdf')
                plt.savefig(os.path.join(self.savepath,plot_name+'.pdf'), format='pdf')
            if show_plots:
                plt.show()

    def clear_plots(self):
        """ Clears the current cache of plots but keeps other settings
        """
        self.plots = collections.OrderedDict()

    def finalize(self):
        """ Finalizes the plotter usage and closes the PdfPages file
        """
        self.pdfpages_file.close()
        self.is_finalized = True
