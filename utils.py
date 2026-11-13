import pandas as pd
import numpy as np
import math as mt
import os
from copy import deepcopy
# from source.filters import *
from scipy.stats import shapiro, normaltest
import seaborn as sbn
import matplotlib.pyplot as plt


def alignIndex(target, predictors):
    predictors.index.name = 'CL-ID'
    return predictors.index.intersection(target["CL-ID"])


def exportData(target, predictors, GUI=False,
               out_folder=None, out_key=None):
    target.index.name = "CL-ID"
    predictors.index.name = "CL-ID"
    if GUI:
        pd.concat([target, predictors], axis=1).to_csv("{}/{}.csv".format(out_folder, out_key))
    else:
        target.to_csv("{}/{}.csv".format(out_folder, out_key))




def normalityTest(data, method='best', n=3):
    res = pd.DataFrame(columns=['shapiro', 'dagostino', 'mean', 'min'],
                       index=data.columns)

    for expi in res.index:
        _, res.loc[expi, 'shapiro'] = shapiro(data.loc[:, expi].dropna())
        _, res.loc[expi, 'dagostino'] = normaltest(data.loc[:, expi], nan_policy='omit')
        res.loc[expi, 'mean'] = np.mean([res.loc[expi, 'shapiro'], res.loc[expi, 'dagostino']])
        res.loc[expi, 'min'] = np.min([res.loc[expi, 'shapiro'], res.loc[expi, 'dagostino']])

    res = res.sort_values(by='min', ascending=False)
    return res


class TheGreatFilter:
    def __init__(self, M, B, mD, nsc_id=None):
        if nsc_id is None:
            self.M = deepcopy(M)
            self.B = deepcopy(B)
            self.mD = deepcopy(mD)
        else:
            ix = mD.loc['nsc', :] == float(nsc_id)
            self.M = M.loc[:, ix]
            self.B = B.loc[:, ix]
            self.mD = mD.loc[:, ix]
        self.n_dropped = 0
        self.n_alive = len(self.M.columns)
        self.n_experiments = None

    def transform_bin(self, compound, percentile=[20, 80], names=['sensitive', 'average', 'resistant'],
                      special_value=99, override_nscid=None):

        if len(percentile) == 1:
            if percentile[0] == 50:
                names = ['sensitive', 'resistant']
            elif percentile[0] < 50:
                names = ['sensitive', 'rest']
            else:
                names = ['rest', 'resistant']

        percentile_full = [0] + percentile + [100]
        if override_nscid is None:
            ncs_id = self.mD.loc['nsc', compound.name]
        else:
            ncs_id = override_nscid
        dist = self.aggregates[ncs_id]
        cutoffs = np.nanpercentile(dist[dist != special_value], percentile_full)

        binned = deepcopy(compound)
        for c in range(1, len(cutoffs)):
            inds = (compound > cutoffs[c - 1]) & (compound <= cutoffs[c])
            binned[inds] = names[c - 1]
        binned[compound <= cutoffs[0]] = names[0]
        binned[compound >= cutoffs[-1]] = names[-1]
        return binned

    def drop_compound_by_class_size(self, percentile=[25, 75], n=5, special_value=99):
        to_keep = np.full(self.M.columns.size, fill_value=True)
        percentile_full = [0] + percentile + [100]

        for i, j in enumerate(self.M.columns):
            ncs_id = self.mD.loc['nsc', j]
            dist = self.aggregates[ncs_id]
            cutoffs = np.nanpercentile(dist[dist != special_value], percentile_full)

            cl_sizes = []
            for c in range(1, len(cutoffs)):
                cl_sizes.append(sum((self.M.loc[:, j] > cutoffs[c - 1]) & (self.M.loc[:, j] <= cutoffs[c])))

            to_keep[i] = all(~(np.array(cl_sizes) < n))
        # print(ncs_id, cl_sizes, ~(np.array(cl_sizes) < n), to_keep[i])

        self.M = self.M.loc[:, to_keep]
        self.B = self.B.loc[:, to_keep]
        self.mD = self.mD.loc[:, to_keep]
        self.n_alive = sum(to_keep)
        self.n_dropped = sum(~to_keep)
        self.get_sum_compounds()
        return self

    def _aggregate_compound(self, nsc_id, special_value=99):
        idx = self.mD.loc['nsc', :] == nsc_id
        M = self.M.loc[:, idx]
        mD = self.mD.loc[:, idx]

        if special_value is not None:
            for j in mD.columns:
                idj = M.loc[:, j] == mD.loc['concentrations', j]
                M.loc[idj, j] = special_value

        return M.values.flatten()

    def aggregate_compounds(self, topk=None, ncs_ids=None,
                            percentiles=None, special_value=None):
        if topk is not None:
            AGGRs = {k: [] for k in self.n_experiments.index[0:topk]}
        elif ncs_ids is not None:
            AGGRs = {float(k): [] for k in ncs_ids}
        else:
            AGGRs = {k: [] for k in self.n_experiments.index}

        for k in AGGRs.keys():
            # print(k)
            AGGRs[k] = self._aggregate_compound(k, special_value=special_value)
        return AGGRs

    def get_sum_compounds(self):
        mDsort = np.sort(self.mD.loc["nsc", :])
        uc = np.unique(mDsort)
        nExps = pd.DataFrame(0, columns=['count'], index=uc)

        c = 0
        arti = 0
        art = uc[0]
        for i in range(mDsort.size):
            # print(i)
            if mDsort[i] == art:
                c += 1
            else:
                nExps.loc[art] = c
                c = 1
                arti += 1
                art = uc[arti]

        nExps.iloc[-1] += 1

        self.n_experiments = nExps.sort_values(by=['count'], ascending=False)
        return self

    def check_compound(self, nsc_id):
        return self.M.loc[:, self.mD.loc['nsc', :] == float(nsc_id)]

    @staticmethod
    def dcbnn(v, n):
        return sum(v.isna()) <= n

    def drop_compound_by_nnans(self, n=5):
        to_keep = self.M.apply(self.dcbnn, args=[n], axis=0)

        self.M = self.M.loc[:, to_keep]
        self.B = self.B.loc[:, to_keep]
        self.mD = self.mD.loc[:, to_keep]
        self.n_alive = sum(to_keep)
        self.n_dropped = sum(~to_keep)
        self.get_sum_compounds()
        return self

    @staticmethod
    def dcbs(v, n, logk):
        # use with df.apply(axis=0)
        avg = np.nanmean(v)  # Vector mean
        return sum((v >= avg + logk / 2) | (v <= avg - logk / 2)) >= n

    def drop_compound_by_specificity(self, n=10, k=3, base=10):
        logk = mt.log(k, base)
        to_keep = self.M.apply(self.dcbs, args=(n, logk), axis=0)

        self.M = self.M.loc[:, to_keep]
        self.B = self.B.loc[:, to_keep]
        self.mD = self.mD.loc[:, to_keep]
        self.n_alive = sum(to_keep)
        self.n_dropped = sum(~to_keep)
        self.get_sum_compounds()
        return self

    @staticmethod
    def dcbc(v, n, logk, sd):
        # use with df.apply(axis=0)
        logrange = v.max() - v.min()
        if logrange > logk:
            return True

        rounded = np.round(v, decimals=sd)  # consider equal by rounding to desired decimal
        uvc = np.array([sum(rounded == uv) for uv in rounded.unique()])
        return ~((uvc >= n).any())

    def drop_compound_by_constant(self, n=10, k=3, base=10, sig_digits=2):
        logk = mt.log(k, base)
        to_keep = self.M.apply(self.dcbc, args=(n, logk, sig_digits), axis=0)

        self.M = self.M.loc[:, to_keep]
        self.B = self.B.loc[:, to_keep]
        self.mD = self.mD.loc[:, to_keep]
        self.n_alive = sum(to_keep)
        self.n_dropped = sum(~to_keep)
        return self

    def drop_compound_by_max_reached(self):
        to_keep = np.full((len(self.M.columns)), fill_value=True)
        for j, cid in enumerate(self.M.columns):
            to_keep[j] = ~(self.M.loc[:, cid] >= self.mD.loc['concentrations', cid]).any()

        self.M = self.M.loc[:, to_keep]
        self.B = self.B.loc[:, to_keep]
        self.mD = self.mD.loc[:, to_keep]
        self.n_alive = sum(to_keep)
        self.n_dropped = sum(~to_keep)
        return self

    def drop_compound_by_custom_value(self, value):
        to_keep = np.full((len(self.M.columns)), fill_value=True)
        for j, cid in enumerate(self.M.columns):
            to_keep[j] = ~(self.M.loc[:, cid] >= value).any()

        self.M = self.M.loc[:, to_keep]
        self.B = self.B.loc[:, to_keep]
        self.mD = self.mD.loc[:, to_keep]
        self.n_alive = sum(to_keep)
        self.n_dropped = sum(~to_keep)
        self.get_sum_compounds()
        return self

    def drop_experiment(self, by_max=True, by_value=None, drop_all=False):
        if by_max:
            for cid in self.M.columns:
                self.M.loc[self.M.loc[:, cid] >= self.mD.loc['concentrations', cid],
                cid] = np.nan
        if by_value is not None:
            for cid in self.M.columns:
                self.M.loc[self.M.loc[:, cid] >= by_value,
                cid] = np.nan

        if drop_all:
            to_keep = ~(self.M.isna().any())
            self.M = self.M.loc[:, to_keep]
            self.B = self.B.loc[:, to_keep]
            self.mD = self.mD.loc[:, to_keep]
            self.n_alive = sum(to_keep)
            self.n_dropped = sum(~to_keep)

        self.by_max = by_max
        self.by_value = by_value
        self.drop_all = drop_all
        return self

    def drop_cell_lines(self, by_name=None, by_nan=None):
        if by_name is not None:
            self.M = self.M.loc[~by_name, :]
        if by_nan is not None:
            nan_arr = self.M.T.isna().sum() / self.M.shape[1]
            drop_inds = (nan_arr > by_nan).values
            self.M = self.M.loc[~drop_inds, :]
            self.B = self.B.loc[~drop_inds, :]

        self.by_name = by_name
        self.by_nan = by_nan
        return self

    def average_single_compound(self, nsc_id):
        comp_ix = self.mD.loc['nsc', :] == float(nsc_id)
        Mc = self.M.loc[:, comp_ix]
        Bb = self.B.loc[:, comp_ix]

        # M_avg = pd.DataFrame(data=np.full((Mc.index.size, 1), fill_value=np.nan),
        #                      index=Mc.index, columns=[nsc_id])

        M_avg = np.nansum(Mc * Bb, axis=1) / np.nansum(Bb, axis=1)
        M_avg = pd.Series(data=M_avg, name=nsc_id, index=Mc.index)

        return M_avg

    def average_by_compound(self):
        np.seterr(invalid='ignore')
        unique_comp = [str(int(uc)) for uc in self.mD.loc["nsc", :].unique()]

        M_avg = pd.DataFrame(data=np.full((self.M.index.size, len(unique_comp)), fill_value=np.nan),
                             index=self.M.index, columns=unique_comp)
        for i, uc in enumerate(unique_comp):
            comp_ix = self.mD.loc['nsc', :] == float(uc)
            Mc = self.M.loc[:, comp_ix]
            Bb = self.B.loc[:, comp_ix]

            M_avg[uc] = np.nansum(Mc * Bb, axis=1) / np.nansum(Bb, axis=1)
            if i % 1000 == 0:
                print('----> Processed {}/{} ...'.format(i, len(unique_comp)))

        M_avg.replace(0, np.nan, inplace=True)  # WARNING: assumes that maximum value is negative!!
        self.TargetAVG = M_avg
        np.seterr(invalid='warn')
        return M_avg

    def export(self, folder, key):
        self.TargetAVG.to_csv(os.path.join(folder, "{}.csv".format(key)))
        out_content = "--- Experiment Args ---\n" \
                      "Remove if maximum concentration reached (by_max):\t{}\n" \
                      "Remove concentration greater than (by_value):\t{}\n" \
                      "Drop columns if NaN present (drop_all):\t{}\n\n" \
                      "--- Cell-line Args ---\n" \
                      "Drop specific cell-line (by_name):\t{}\n" \
                      "Drop cell-lines if NaN percentage exceeds (by_nan):\t{}".format(self.by_max, self.by_value,
                                                                                       self.drop_all,
                                                                                       self.by_name, self.by_nan)
        os.system("echo '{}' > {}.out".format(out_content,
                                              os.path.join(folder, key)))