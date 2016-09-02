import numpy as np
import matplotlib.pyplot as plt
from sklearn_utils.utils.plotting import plot_steps_with_errors
import utils.statistics
import utils.selection

class PlotComponent:
    def __init__(self, name, data, observable):
        self.name       = name
        self.data       = data
        self.observable = observable

    def apply_selection(self, selection, print_efficiencies=True, print_single_cut_efficiencies=True):
        if type(selection) == type({}): # Ramon's selection style with dicts
            if selection != {}:
                selectionorder=list(selection.keys())
                self.data = utils.selection.apply_selection_to_dataframe(self.data, selection, selectionorder,
                                                                         print_efficiencies=print_efficiencies,
                                                                         print_single_cut_efficiencies=print_single_cut_efficiencies)
        elif type(selection) == type(""):
            print("WARNING: Using experimental string selection style")
            print("WARNING: {}".format(selection))
            self.data = self.data.query(selection)
        else:
            print("ERROR: Unsupported selection type")

class Plot:
    def __init__(self, title):
        self.components  = {}
        self.title       = title
        self.observables = []
        self.x_min       = None
        self.x_max       = None

    def add_component(self, data, observable, component_name=""):
        if component_name=="":
            component_name = str(len(self.components))
        self.components[component_name] = PlotComponent(component_name, data, observable)
        self.observables.append(observable)
        x_min = data[observable].min()
        x_max = data[observable].max()
        if (not self.x_min) or (x_min < self.x_min):
            self.x_min = x_min
        if (not self.x_max) or (x_max > self.x_max):
            self.x_max = x_max

    def apply_selection(self, selection, print_efficiencies=True, print_single_cut_efficiencies=True):
        for component_name, component in self.components.items():

            if print_efficiencies or print_single_cut_efficiencies:
                print("Plot: {:<15}  Dataset: {:<15}".format(self.title, component.name))
                print("-----------------------------------------------------------------")

            if type(selection) == type({}): # Ramon's selection style with dicts
                if selection != {}:
                    if type(list(selection.values())[0]) == type({}): #Then we need to get the component first
                        if component.name in selection:
                            selection = selection[component_identifier]
                        else:
                            print("No selection for component {}".format(component_identifier))
                    else: #Apply same selection for all components
                        print("Will apply same selection to all components")
                        pass
                
            component.apply_selection(selection)

    def print_components(self):
        print("Plot: {}".format(self.title))
        print("{:<10} {:<10} {:<10}".format("component", "rows", "columns"))
        for component in self.components:
            print("{:<10} {:<10} {:<10}".format(component.name, len(component.data),
                                                len(component.data.columns)))

    def plot(self):
        for component_name, component in self.components.items():
            y, bins = np.histogram(component.data[component.observable].values, bins=100, range=(self.x_min,self.x_max))
            errors = utils.statistics.poissonian_cls(y)
            y, errors = utils.statistics.normalize_histogram(y, errors)
            plot_steps_with_errors(bins, y, errors, label=component.name, alpha=0.1)

        plt.legend(loc='best')
        fig = plt.gcf()
        plt.xlabel(set(self.observables))
        plt.title(self.title)

        fig.set_size_inches(8,6)

    def copy(self):
        plot_copy = Plot(self.title)
        import copy
        plot_copy.components = copy.deepcopy(self.components)
        plot_copy.observables = self.observables.copy()
        plot_copy.x_min = self.x_min
        plot_copy.x_max = self.x_max
        return plot_copy

class Plotter:
    def __init__(self):
        self.plots = {}

    def create_plot(self, plot_name, datasets, observable, component_labels=[]):
        """ Creates single plot of an observables in multiple datasets
        """
        if not type(datasets)==type([]):
            datasets = [datasets]
        
        if component_labels and len(component_labels) != len(datasets):
            raise Exception('length of component_labels does not match number of components')

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

        for plot_name, observable in zip(plot_names, observables):
            self.create_plot(plot_name, datasets, observable, component_labels)

    def duplicate_plot(self, old_plot_name, new_plot_name, selection=None):
        """ Duplicates plot and applies a selection to the copy
        """
        if old_plot_name in self.plots:
            self.plots[new_plot_name] = self.plots[old_plot_name].copy()
            self.plots[new_plot_name].title = new_plot_name
            if selection:
                self.apply_selection_to_plot(new_plot_name, selection)
        else:
            print("ERROR: couldn't duplicate because {} was not found.".format(old_plot_name))

    def duplicate_plots(self, suffix_to_append="copy", plots=[], selection=None):
        """ Duplicates multiple plots and applies a selection to the copy
        """
        if plots == []:
            plots = list(self.plots.keys())

        for old_plot_name in plots:
            new_plot_name = old_plot_name+ "_" + suffix_to_append
            self.duplicate_plot(old_plot_name, new_plot_name, selection)

    def apply_selection_to_plot(self, plot_name, selection):
        """ Applies selection to a single plot
        """
        if plot_name in self.plots:
            self.plots[plot_name].apply_selection(selection)
        else:
            print("Plotter::apply_selection_to_plot couldn't apply selection plot {} not found".format(plot_name))

    def apply_selection_to_plots(self, selection):
        """ Applies selection to all plots
        """
        for plot_name in self.plots:
            self.plots[plot_name].apply_selection(selection)

    def print_plots(self):
        """ Printout of plots with their components
        """
        for plot_name in self.plots:
            plot.print_components()

    def plot(self, show_plots=True):
        """ Plot all plots
        """
        for plot_name in self.plots:
            self.plots[plot_name].plot()
            if show_plots:
                plt.show()