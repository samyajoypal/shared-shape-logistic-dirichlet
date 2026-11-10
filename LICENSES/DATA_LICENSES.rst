Dataset licenses and attribution
===============================

Third-party data retain their upstream terms. No repository code notice
replaces these terms. The only datasets bundled with this repository are
Abalone and Seeds from the UCI Machine Learning Repository. Other datasets
are obtained separately by the documented download and preparation steps.
License information below was checked on 10 November 2026.

Bundled UCI data: Creative Commons Attribution 4.0
-------------------------------------------------

Both official UCI dataset pages identify the Creative Commons Attribution
4.0 International license (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/ . Its legal text is at
https://creativecommons.org/licenses/by/4.0/legalcode . Redistribution
must retain appropriate credit, the license link and an indication of
changes. Neither the dataset creators nor UCI endorse this research.

Abalone
~~~~~~~

Nash, W., Sellers, T., Talbot, S., Cawthorn, A., and Ford, W. (1994).
Abalone [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C55C7W .

Source and license notice: https://archive.ics.uci.edu/dataset/1/abalone .
Original download: https://archive.ics.uci.edu/static/public/1/abalone.zip .

The bundled source records are unchanged. The prepared analysis table
adds original one-based row identifiers and group metadata, retains the
published sex/infant category as an external label, and selects Length,
Diameter, Height and ShuckedWeight. Original rows 1258 and 3997 are
excluded from that prepared table because Height is exactly zero and its
logarithm is undefined. All other positive records, including unusual
measurements, remain. The resulting table has 4,175 rows; the original
source has 4,177. Published numerical values and their recorded scaling
are retained; no unit back-conversion, imputation or fitted preprocessing
is applied by data preparation. Analysis transformations are recorded
separately in the study configuration.

Seeds
~~~~~

Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P., and
Lukasik, S. (2010). Seeds [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5H30K .

Source and license notice: https://archive.ics.uci.edu/dataset/236/seeds .
Original download: https://archive.ics.uci.edu/static/public/236/seeds.zip .

The original whitespace-separated source is retained. Its prepared CSV
adds column names and preserves all 210 records, numerical values, row
order, seven measured/derived features and the published variety code.
The benchmark analysis excludes compactness from the fitted coordinates
because it is derived from area and perimeter; this does not delete the
column from the source table. Variety codes are external labels and are
not used as mixture-fitting inputs.

Separately downloaded sources
-----------------------------

No source or prepared measurement bytes from the following five datasets
are bundled. The downloader checks the study's recorded source hashes;
conversion and analysis settings are documented separately. Upstream
package licenses are reported for provenance, not replaced with CC BY
or with the repository's code notice.

* Olive oil: ``pdfCluster::oliveoil``, attributed by its help to Forina,
  Lanteri, Armanino, Casolino, Casale and Oliveri (2008), V-PARVUS,
  Universita di Genova. The enclosing pdfCluster package is GPL-2.
  Dataset help:
  https://search.r-project.org/CRAN/refmans/pdfCluster/html/oliveoil.html .
  Package terms: https://CRAN.R-project.org/package=pdfCluster .
  Conversion preserves 572 rows; the fixed three- and six-acid selections
  and oleic reference are specified by the application configurations.
* Crabs: ``MASS::crabs``. Campbell, N. A., and Mahon, R. J. (1974).
  A multivariate study of variation in two species of rock crab of genus
  Leptograpsus. Australian Journal of Zoology 22, 417--425.
  https://doi.org/10.1071/ZO9740417 . The MASS package offers GPL-2 or GPL-3.
  Dataset help:
  https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/crabs.html .
  Conversion preserves 200 rows, the five millimetre measurements, and
  the recorded colour-form and sex labels.
* Hydrochem: ``compositions::Hydrochem``. Otero, N., Tolosana-Delgado, R.,
  Soler, A., Pawlowsky-Glahn, V., and Canals, A. (2005). Relative vs.
  absolute statistical analysis of compositions: a comparative study of
  surface waters of a Mediterranean river. Water Research 39, 1404--1414.
  https://doi.org/10.1016/j.watres.2005.01.012 . The data help credits
  Raimon Tolosana-Delgado and contains its own version-2-or-newer reuse
  notice; the compositions package states GPL (>= 2).
  https://raw.githubusercontent.com/cran/compositions/master/man/Hydrochem.Rd .
  Preparation selects Na, K, Mg and Ca and retains all 485 rows and the
  30 recorded site IDs, without unit conversion.
* GEMAS: ``robCompositions::gemas``. GEMAS is a cooperation between the
  EuroGeoSurveys Geochemistry Expert Group and Eurometaux; R integration
  is credited to Peter Filzmoser and Matthias Templ. Reference: Reimann,
  C., Birke, M., Demetriades, A., Filzmoser, P., and O'Connor, P. (eds.),
  2014, Chemistry of Europe's agricultural soils, Parts A and B,
  Geologisches Jahrbuch B102/B103, Schweizerbart. The robCompositions
  package states GPL (>= 2). Dataset help:
  https://search.r-project.org/CRAN/refmans/robCompositions/html/gemas.html .
  Project: https://gemas.eurogeosurveys.org/ . Preparation selects
  Ca, Fe, K, Mg, Si and Al, preserving 2,108 rows, country codes and
  soil-class metadata, including blank soil-class values.
* Faithful: ``datasets::faithful`` from R, credited to W. Hardle;
  Hardle (1991), Smoothing Techniques: With Implementation in S.
  R's datasets package is part of R; the source distribution supplies
  the GNU General Public License version 2. Dataset help:
  https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html .
  R license: https://svn.r-project.org/R/trunk/COPYING . The extraction
  preserves the exact 272-row version, rounding and two time coordinates
  of that source rather than substituting another geyser series.

Acquiring a dataset does not transfer its ownership. Any later
redistribution of the separately downloaded data must preserve applicable
upstream notices and terms.
