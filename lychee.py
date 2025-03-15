import numpy

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from tangermeme.seqlet import recursive_seqlets
from tangermeme.tools.tomtom import tomtom


def lychee(X_attr, n_exemplars=100, n_components=20, seqlet_threshold=0.01, 
	min_seqlet_len=4, cluster_kwds={'min_cluster_size': 100, 'n_jobs': -1},
	additional_flanks=3, random_state=None):
	"""An experimental method for motif discovery from attributions.
	
	The method process in three stages. First, call seqlets using the new
	recursive seqlet caller. Second, calculate similarity scores between a random
	subset and all seqlets using TOMTOM. Third, use HDBSCAN to cluster seqlets
	and extract the patterns using the already-calculated alignments.
	
	
	Parameters
	----------
	X_attr: torch.Tensor, shape=(n, 4, seq_len)
		A tensor of attributions, which can be from any method but are likely from
		deep_lift_shap. Attributions must be provided for only the observed
		nucleotide, regardless of which method is used.
	
	n_exemplars: int, optional
		The size of the subset to use when calculating kernel representations for
		each seqlet. As this number gets higher, the likelihood of discovering rarer
		patterns increases but memory and compute cost also scale linearly. Default
		is 100.
	
	n_components: int, optional
		The number of components to use in the PCA projection of the kernel
		representations. This should be roughly 3-5x the number of expected patterns
		in the data. This is roughly linearly correlated with speed. Default is 20.
	
	seqlet_threshold: float, optional
		The p-value threshold to use for seqlet discovery by the recursive seqlet
		method. Default is 0.01.
	
	min_seqlet_len: int, optional
		The minimum length of the seqlets. This should be as high as possible without
		being too long to discard motifs. Default is 4.
	
	additional_flanks: int, optional
		Additional bp to add to either side of each seqlet after discovery. This is
		not used in the seqlet discovery process, but may cause some seqlets too
		near the edges of the seqlets to be discarded. Seqlets will not overlap in
		their core regions but may overlap in these flanking regions. Default is
		3.
	
	cluster_kwds: dict, optional
		Arguments to pass into the clustering method.
	
	random_state: int or None
		A random state to use for determinism. None means not deterministic. Default
		is None.
		
	
	Returns
	-------
	cwms: list of numpy.ndarrays
		A list of patterns derived from seqlets that are in the same cluster and are
		then aligned to each other. CWMs are the average attribution value at each
		position: specifically, the sum of the attribution across all seqlets (0s
		if the seqlets do not align at that position) divided by the total number of
		seqlets.
	
	pwms: list of numpy.ndarrays
		A list of patterns derived from seqlets that are in the same cluster and are
		then aligned to each other. PWMs are the average frequency of each character
		agnostic of the attribution value.
	
	seqlets: pandas.DataFrame
		The list of called seqlets used in the subsequent steps.
	
	cluster_idxs: numpy.ndarray
		The cluster index for each seqlet. Negative values refer to outliers, and
		not to just another cluster.
	"""
	
	# Calculate seqlets using the tangermeme 
	seqlets = recursive_seqlets(X_attr.sum(dim=1), threshold=seqlet_threshold, 
		min_seqlet_len=min_seqlet_len, additional_flanks=additional_flanks)

	Xs = []
	for _, (idx, start, end) in seqlets[['example_idx', 'start', 'end']].iterrows():
		X_ = X_attr[idx, :, start:end].numpy(force=True)
		X_ = X_ / abs(X_).max()
		Xs.append(X_)

	seqlet_lengths = numpy.array([x.shape[-1] for x in Xs])
	max_seqlet_length = max(seqlet_lengths)

	random_state = numpy.random.RandomState(random_state)
	exemplar_idxs = random_state.choice(len(Xs), replace=False, size=n_exemplars)
	
	X_exemplars = [Xs[idx] for idx in exemplar_idxs]

	p_values, _, offsets, _, strands = tomtom(X_exemplars, Xs)
	p_values = numpy.log(p_values + 1e-10).T
	p_values_pca = PCA(n_components=n_components).fit_transform(p_values)
	
	clusterer = HDBSCAN(**cluster_kwds)
	cluster_idxs = clusterer.fit_predict(p_values_pca)
	
	cwms, pwms = [], []

	for cluster_idx in numpy.unique(cluster_idxs):
		if cluster_idx < 0:
			continue
			
		seqlet_idxs = numpy.where(cluster_idxs == cluster_idx)[0]

		best_idx = p_values[seqlet_idxs].mean(axis=0).argmin()

		agg_max_len = 2*max_seqlet_length + seqlet_lengths[best_idx]-2
		agg_score = numpy.zeros((4, agg_max_len))
		agg_count = numpy.zeros((4, agg_max_len))

		for seqlet_idx in seqlet_idxs:
			start = int(max_seqlet_length - offsets[best_idx, seqlet_idx])
			end = start + seqlet_lengths[seqlet_idx]

			x = Xs[seqlet_idx]
			if strands[best_idx, seqlet_idx] == 1:
				x = x[::-1, ::-1]

			agg_score[:, start:end] += x
			agg_count[:, start:end] += (x != 0)

		flanks = agg_count.sum(axis=0) == 0

		agg_score = agg_score[:, ~flanks]
		agg_count = agg_count[:, ~flanks]

		cwm = (agg_score + 1e-9) / len(seqlet_idxs) # (agg_count + 1e-6)
		cwms.append(cwm)

		pwm = agg_count / agg_count.sum(axis=0)
		pwms.append(pwm)
	
	return cwms, pwms, seqlets, cluster_idxs

