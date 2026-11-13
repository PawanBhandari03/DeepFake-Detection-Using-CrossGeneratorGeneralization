"""Link targets for the reference list of the research paper (key = key in build_full_paper.REFS).

Every arXiv link and DOI below was checked on 13 Nov 2026: the title returned by the arXiv API or by Crossref
matches the title in our reference list. The other links were opened and checked by hand-written rules:
  qu       - the CVPR 2023 open-access page shows the exact title
  krawetz  - the Black Hat white paper PDF (43 pages) opens with the title 'A Picture's Worth... Digital Image Analysis and Forensics'
  naeini   - Crossref DOI of the AAAI 2015 paper (title matches)
  pedregosa- JMLR page (title matches)
  c2pa     - the specification page named in the reference itself
  platt    - Platt's chapter is part of the book 'Advances in Large-Margin Classifiers' (MIT Press); no separate DOI exists,
             so the link goes to the book's DOI (Crossref confirms the book)
"""
REF_URLS = {
    "rombach": "https://arxiv.org/abs/2112.10752",
    "rossler": "https://arxiv.org/abs/1901.08971",
    "ojha": "https://arxiv.org/abs/2302.10174",
    "wang": "https://arxiv.org/abs/1912.11035",
    "grommelt": "https://arxiv.org/abs/2403.17608",
    "corvi": "https://arxiv.org/abs/2211.00680",
    "forensichub": "https://arxiv.org/abs/2505.11003",
    "radford": "https://arxiv.org/abs/2103.00020",
    "zhu": "https://arxiv.org/abs/2306.08571",
    "livernoche": "https://arxiv.org/abs/2509.09495",
    "li2020": "https://arxiv.org/abs/1909.12962",
    "zhang": "https://arxiv.org/abs/1604.02878",
    "qu": "https://openaccess.thecvf.com/content/CVPR2023/html/Qu_Towards_Robust_Tampered_Text_Detection_in_Document_Image_New_Dataset_CVPR_2023_paper.html",
    "zhou": "https://arxiv.org/abs/1805.04953",
    "fridrich": "https://doi.org/10.1109/TIFS.2012.2190402",
    "krawetz": "https://www.blackhat.com/presentations/bh-dc-08/Krawetz/Whitepaper/bh-dc-08-krawetz-WP.pdf",
    "ronneberger": "https://arxiv.org/abs/1505.04597",
    "milletari": "https://arxiv.org/abs/1606.04797",
    "guo": "https://arxiv.org/abs/1706.04599",
    "platt": "https://doi.org/10.7551/mitpress/1113.001.0001",
    "geifman": "https://arxiv.org/abs/1705.08500",
    "tan": "https://arxiv.org/abs/1905.11946",
    "c2pa": "https://c2pa.org/specifications/",
    "paszke": "https://arxiv.org/abs/1912.01703",
    "pedregosa": "https://jmlr.org/papers/v12/pedregosa11a.html",
    "ci": "https://arxiv.org/abs/2511.19111",
    "lilyu": "https://arxiv.org/abs/1811.00656",
    "smith": "https://doi.org/10.1109/ICDAR.2007.4376991",
    "naeini": "https://doi.org/10.1609/aaai.v29i1.9602",
    "gptedit": "https://arxiv.org/abs/2507.21033",
    "fawcett": "https://doi.org/10.1016/j.patrec.2005.10.010",
    "efron": "https://doi.org/10.1214/aos/1176344552",
}
