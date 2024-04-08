# IsoSigPGML

Repository containing code for generation, visualisation, and network analysis of Pachner graphs representing triangulations of 3-manifolds. The triangulations are represented as IsoSigs, and supervised machine learning methods are shown to be able to differentiate manifolds directly from this highly compressed representation.      

------------------------------------------------------------------------
The folder `Data` contains the generated graphs and IsoSig databases used in the machine learning experiments discussed in the paper. The deeper Pachner graphs analysed have been stored as compressed files and should be unzipped in place before further analysis.    

The scripts demonstrate functionality for repeating the experiments discussed, including direct classification of manifolds from their triangulation IsoSigs with neural networks (`DirectClassification.py`), and then further scripts for gradient saliency analysis of these trained networks (`Saliency.py`), and for the equivalent use of a BERT transformer model (`Transformer.py`). Further scripts generate Pachner graphs and the respective IsoSig data (`PG_Generator.py`), then perform relevant network analysis (`PG_NA.py`). There is an equivalent script for network analysis of Pachner graphs for 3-manifolds in the snappy orientable cusped census (`PG_NA_OCC.py`). Finally, statistics and figures are created in `PGNA_Analysis.py`.    

An introductory notebook, aptly named `IntroductoryNotebook.ipynb`, demonstrates the broad methodology through Pachner graph generation, network analysis, and provides a variety of ways to visualise these graphs.    

------------------------------------------------------------------------
# BibTeX Citation
raise NotImplementedError
