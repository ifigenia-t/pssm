class Result:
    def __init__(
        self,
        comparison_results,
        similarity,
        gini_1,
        gini_2,
        base_name="",
        elm="",
        consensus="",
        quality="",
        motif_1="",
        motif_2="",
        kendalls="",
        spearmans="",
        dots="",
        ssds="",
        kls="",
    ):
        self.elm = elm
        self.base_name = base_name
        self.quality = quality
        self.comparison_results = comparison_results
        self.comparison_results_kendalls = kendalls
        self.comparison_results_spearmans = spearmans
        self.comparison_results_dots = dots
        self.comparison_results_ssds = ssds
        self.comparison_results_kls = kls
        self.motif_1 = motif_1
        self.motif_2 = motif_2
        self.consensus = consensus
        self.gini_1 = gini_1
        self.gini_2 = gini_2
        self.similarity = similarity
