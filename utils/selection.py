def apply_cut_to_dataframe(dataframe, dep, cutrange, ignore_missing_columns=False):
  fr_cut = dataframe
  formatstring = "{:<40} {:<5.4}"
  if dep not in fr_cut.columns:
    if not ignore_missing_columns:
      print("ERROR {} is not in fr_cut columns - returning original fr_cut".format(dep))
      return fr_cut
    else:
      print("WARNING {} is not in fr_cut columns - skipping cut".format(dep))
  if cutrange[1] is None:
    fr_cut = fr_cut[(fr_cut[dep]>=cutrange[0])]
  elif cutrange[0] is None:
    fr_cut = fr_cut[(fr_cut[dep]<=cutrange[1])]
  else:
    fr_cut = fr_cut[(fr_cut[dep]>=cutrange[0]) & (fr_cut[dep]<=cutrange[1])]
  return fr_cut


def apply_selection_to_dataframe(dataframe, selection, selectionorder=[], print_efficiencies=False,
                                 print_single_cut_efficiencies=False, warn_selection_order=False,
                                 ignore_missing_columns=False):
  if selection=={}:
    print("WARNING no selection given - returning original dataframe")
    return dataframe

  from more_itertools import unique_everseen

  if print_single_cut_efficiencies:
    formatstring = "{:<50} {:<10} {:<10} {:<10}"
    print(formatstring.format("Cut", "Fraction", "Rel Fract", "Single"))
    formatstring = "{:<50} {:<10.4} {:<10.4} {:<10.4}"
  elif print_efficiencies:
    formatstring = "{:<50} {:<10} {:<10}"
    print(formatstring.format("Cut", "Fraction", "Rel Fract"))
    formatstring = "{:<50} {:<10.4} {:<10.4}"

  fr_cut = dataframe

  if selectionorder == []:
    observables = selection.keys()
  else:
    observables = selectionorder.copy()
    # might not contain all observables in the selection
    keys = list(selection.keys())
    observables += keys
    observables = list(unique_everseen(observables)) #removes duplicates while preserving order

  last_eff = None
  for dep in observables:
    if dep in selection:
      cutrange = selection[dep]
      last_eff = 100 if not last_eff else last_eff
      if cutrange[1] is None:
        cutstring = "{} >={}".format(dep, cutrange[0])
      elif cutrange[0] is None:
        cutstring = "{} <={}".format(dep, cutrange[1])
      else:
        cutstring = "{}<= {} <={}".format(cutrange[0], dep, cutrange[1])
      fr_cut = apply_cut_to_dataframe(fr_cut, dep, cutrange, ignore_missing_columns)
      if print_single_cut_efficiencies:
        fr_cut_for_single = apply_cut_to_dataframe(dataframe, dep, cutrange, ignore_missing_columns)
        single_eff = len(fr_cut_for_single.index)/len(dataframe.index)*100
      cumulative_eff = len(fr_cut.index)/len(dataframe.index)*100

      rel_eff = cumulative_eff/last_eff*100 if last_eff else cumulative_eff
      last_eff = cumulative_eff

      if print_single_cut_efficiencies:
        print(formatstring.format(cutstring, cumulative_eff, rel_eff, single_eff))
      elif print_efficiencies:
        print(formatstring.format(cutstring, cumulative_eff, rel_eff))

    elif warn_selection_order:
      print("WARNING: {} is not part of the selection".format(dep))
  return fr_cut
