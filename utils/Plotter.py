import numpy as np
import matplotlib.pyplot as plt
from sklearn_utils.utils.plotting import plot_steps_with_errors
import utils.statistics
import utils.selection

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

        self.dict_selection   = {}
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
                raise Exception("String like selection is already set - it is unclear how to combine {} with {} \
                    for plot {} component {}".format(selection, self.string_selection, self.mother_name, self.name))
        else:
            raise Exception("Unknown selection type")


    def apply_selection(self, print_efficiencies=False, print_single_cut_efficiencies=False):
        
        if self.dict_selection != {}:
            selectionorder = list(self.dict_selection.keys())
            self.data = utils.selection.apply_selection_to_dataframe(self.data, self.dict_selection, selectionorder,
                                                                     print_efficiencies=print_efficiencies,
                                                                     print_single_cut_efficiencies=print_single_cut_efficiencies)
        if self.string_selection != "":
            print("WARNING: Using experimental string selection style for {}".format(self.string_selection))
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
        self.components        = {}
        self.title             = title
        self.observables       = []
        self.x_min             = None
        self.x_max             = None

        self.range_auto_update       = True
        self.range_part_of_selection = False

        self.is_selected       = False


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


    def apply_selection(self, print_efficiencies=False, print_single_cut_efficiencies=False):
        if not self.range_auto_update and not self.range_part_of_selection:
            print("WARNING: Range for plot {} has been manually set.".format(self.title))
            print("WARNING: Efficiencies are calculated against the full complete dataset.")
        for component_name, component in self.components.items():
            if print_efficiencies or print_single_cut_efficiencies:
                if(self.is_selected):
                    print("WARNING {} {} has been selected before. Efficiencies are not normalized to the total data".format(self.title,
                                                                                                                             component_name))
                print("Appplying selections for Plot: {:<15}  Dataset: {:<15}".format(self.title, component.name))
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
        print("WARNING: you updated the range of {} by hand - won't change it.".format(self.title))



    def print_components(self):
        print("Plot: {}".format(self.title))
        print("{:<10} {:<10} {:<10}".format("component", "rows", "columns"))
        for component in self.components:
            print("{:<10} {:<10} {:<10}".format(component.name, len(component.data),
                                                len(component.data.columns)))


    def plot(self, verbose=False):
        if verbose:
            self.apply_selection(print_efficiencies=True,
                                 print_single_cut_efficiencies=True)
        else:
            self.apply_selection()

        for component_name, component in self.components.items():
            if self.range_auto_update:
                x_min = component.get_min()
                x_max = component.get_max()
                if (not self.x_min) or (x_min < self.x_min):
                    self.x_min = x_min
                if (not self.x_max) or (x_max > self.x_max):
                    self.x_max = x_max

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
        
        plot_copy.components              = copy.deepcopy(self.components)
        plot_copy.observables             = self.observables.copy()
        plot_copy.x_min                   = self.x_min
        plot_copy.x_max                   = self.x_max
        plot_copy.range_auto_update       = self.range_auto_update
        plot_copy.range_part_of_selection = self.range_part_of_selection
        plot_copy.is_selected             = self.is_selected

        return plot_copy



class Plotter:
    def __init__(self):
        self.plots = {}


    def __getitem__(self, plot_name):
        return self.plots[plot_name]


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
        sorted_names = list(self.plots.keys())
        sorted_names.sort()

        for plot_name in sorted_names:
            self.plots[plot_name].plot(verbose)
            if show_plots:
                plt.show()