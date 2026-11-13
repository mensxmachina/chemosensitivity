import pandas as pd
from utils import *
import pickle
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")


# ====== PARAMS ======
MAXVAL = -1
NANALLOWED = 10
N = 10
FC = 10
CUTOFF = [25, 75]
MINPERCLASS = 10

SPECIAL_VALUE = None
PRECOMPUTE = False  # Compute or load tgf instance
onlyAVG = True  # Compute only averaged experiments

DATA_FOLDER = './data'
TARGET_FOLDER = './targets'
PRECOMP_FOLDER = './precomp'
# ====================


if PRECOMPUTE:
	# Load Concentrations, etc  (independent from predictor dataset)
	M = pd.read_csv(os.path.join(DATA_FOLDER, 'outcomes.csv'), index_col=0)
	B = pd.read_csv(os.path.join(DATA_FOLDER, 'outcomes_bool.csv'), index_col=0)
	mD = pd.read_csv(os.path.join(DATA_FOLDER, 'outcomes_metadata.csv'), index_col=0)

	"""
	STEP 0: INITIALIZATION
	"""
	# Init
	initfile = "tgf-init.pcl"
	if not os.path.exists(os.path.join(PRECOMP_FOLDER, initfile)):
		tgf = TheGreatFilter(M, B, mD)

		# Aggregate all experiments
		tgf.get_sum_compounds()  # Rank exps by count before filtering
		tgf.aggregates = tgf.aggregate_compounds(special_value=SPECIAL_VALUE)  # Get distribution for each compound

		# Save init:
		with open(os.path.join(PRECOMP_FOLDER, initfile), 'wb') as f:
			pickle.dump(tgf, f)
	else:
		with open(os.path.join(PRECOMP_FOLDER, initfile), 'rb') as f:
			tgf = pickle.load(f)
	print("nExps:\t{}".format(tgf.n_alive))
	print("Unique nExps (Compounds):\t{}".format(tgf.n_experiments.size))
	tgf_init = deepcopy(tgf)

	"""
	STEP 1: No GI50 value (log10 M) > `MAXVAL`
	"""
	# Drop exps by specific value
	print("\n -- Specific value")
	tgf.drop_compound_by_custom_value(value=MAXVAL)
	print("Alive:\t{}\nDropped:\t{}".format(tgf.n_alive, tgf.n_dropped))
	print("Unique nExps (Compounds):\t{}".format(tgf.n_experiments.size))


	"""
	STEP 2: At most `NANALLOWED` missing values
	"""
	# Drop exps with n > NANALLOWED
	print("\n -- Nans in target")
	tgf.drop_compound_by_nnans(n=NANALLOWED)
	# Checks
	print("Alive:\t{}\nDropped:\t{}".format(tgf.n_alive, tgf.n_dropped))
	print("Unique nExps (Compounds):\t{}".format(tgf.n_experiments.size))


	"""
	STEP 3: At least `N` cell lines differing from the experiment mean by at least `FC`-fold (log10(`FC`) units)
	"""
	# Drop exps if n<N with foldchange < FC
	print("\n -- Specificity")
	tgf.drop_compound_by_specificity(n=N, k=FC, base=10)
	# Checks
	print("Alive:\t{}\nDropped:\t{}".format(tgf.n_alive, tgf.n_dropped))
	print("Unique nExps (Compounds):\t{}".format(tgf.n_experiments.size))


	"""
	STEP 4: At least `MINPERCLASS` cell lines per class
	(sensitive, average, resistant) - sensitive <`CUTOFF[0]`%, resistant > `CUTOFF[1]`%
	"""
	# Drop exps with min samples per class after cutoff
	print("\n -- Cutoff min class")
	tgf.drop_compound_by_class_size(percentile=CUTOFF, n=MINPERCLASS)
	# Checks
	print("Alive:\t{}\nDropped:\t{}".format(tgf.n_alive, tgf.n_dropped))
	print("Unique nExps (Compounds):\t{}".format(tgf.n_experiments.size))

	# CHECK - Recompute top compounds
	tgf.get_sum_compounds()

	# Export
	path = os.path.join(PRECOMP_FOLDER,
						'filter_max{}_N{}_FC{}_nnan{}_CO{}-{}>{}.pcl'.format(abs(MAXVAL), N, FC, NANALLOWED,
	                                                                         CUTOFF[0], CUTOFF[1] if len(CUTOFF) > 1 else 'binary',
	                                                                         MINPERCLASS))
	with open(path, 'wb') as f:
		pickle.dump(tgf, f)
else:
	path = os.path.join(PRECOMP_FOLDER,
						'filter_max{}_N{}_FC{}_nnan{}_CO{}-{}>{}.pcl'.format(abs(MAXVAL), N, FC, NANALLOWED,
																			 CUTOFF[0], CUTOFF[1] if len(CUTOFF) > 1 else 'binary',
																			 MINPERCLASS))
	with open(path, 'rb') as f:
		tgf = pickle.load(f)



# GENERATE TARGETS
nscIDs = tgf.n_experiments.index
cell_line_names = pd.read_csv(os.path.join(DATA_FOLDER, 'outcomes.csv'), usecols=[0]).values.flatten().tolist()

TARGETS = pd.DataFrame(index=cell_line_names)
TARGETS_num = deepcopy(TARGETS)
for ti, nsci in enumerate(nscIDs):
	TARGETS.loc[cell_line_names, "{}_averaged".format(int(nsci))] = \
		tgf.transform_bin(tgf.average_single_compound(nsc_id=nsci), percentile=CUTOFF, override_nscid=nsci)

	TARGETS_num.loc[cell_line_names, "{}_averaged".format(int(nsci))] = tgf.average_single_compound(nsc_id=nsci)

	if not onlyAVG:
		exps = tgf.check_compound(nsci)
		for expi in exps.columns:
			TARGETS.loc[cell_line_names, "{}_{}".format(int(nsci), expi)] = \
				tgf.transform_bin(tgf.M.loc[:, expi], percentile=CUTOFF)

			TARGETS_num.loc[cell_line_names, "{}_{}".format(int(nsci), expi)] = tgf.M.loc[:, expi]
	if (ti+1) % 500 == 0:
		print(f"Calculated {ti+1}/{nscIDs.size} targets")

"""
STEP 5: Final export - only averaged
"""
# FINAL EXPORT
TARGETS.to_csv(os.path.join(TARGET_FOLDER,
							'TARGETS-BINNED-{}.csv'.format('' if not onlyAVG else 'averaged')), index=True)
TARGETS_num.to_csv(os.path.join(TARGET_FOLDER,
								'TARGETS-GI50-{}.csv'.format('' if not onlyAVG else 'averaged')), index=True)

print()

