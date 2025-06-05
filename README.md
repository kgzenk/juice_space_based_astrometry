Constraining the Ephemeris and Interior Structure of Io using Space-Based Astrometry by JUICE
==========================

In the spirit of open science, this repository bundles all pieces of code and newly-created data that is necessary to 
recreate the results presented in [https://doi.org/10.1016/j.pss.2025.106112](https://doi.org/10.1016/j.pss.2025.106112).
First, for the sake of completeness, while not providing any new information to the avid reader of the original paper, 
the abstract once again summarises the work's main points. 
Subsequently, we outline any required software packages and additional external data (such as the SPICE kernels of 
JUICE and Cassini) required, followed by a brief description of the functions and purposes of individual files and 
their overall structure.

"Finally, we make some remarks on why linear systems are so important. The answer is simple: because we can solve 
them!" (Richard Feynman, 1963)


Abstract
------------

Being among the most promising candidates for potential extraterrestrial habitats within our Solar System, the Galilean 
satellites are going to be extensively studied by the upcoming JUICE and Europa Clipper missions. Both spacecraft will 
provide radio science tracking data, which will allow the satellites ephemerides to be determined to much greater 
accuracy than is currently the case. Yet, with no flybys of Io, these data sets will be skewed towards the three outer 
satellites. To mitigate this imbalance, optical space-based astrometry from JUICE will provide a valuable contribution.

To quantify the contribution of JUICE astrometry, we have performed the inversion of simulated optical astrometric 
observations by JUICE, using suitable a priori covariance to represent the radio-science-only solution. 
Incorporating the astrometry into the ephemeris solution requires the consideration of the offset between Io's 
centre-of-figure(COF, which astrometry measures) and the centre-of-mass (COM, which the ephemeris solution requires). 
We explicitly account for the offset between COF and COM as an estimated parameter in our model.

We assess the contribution of the optical observations to the ephemeris solution as a function of the radio science 
true-to-formal-error ratio (describing the statistical realism of the simulated radio science solution), as well as 
optical data quantity and planning. From this, we discuss to which extent space-based astrometry could help to validate
the radio science solution, and under which conditions the data could improve the orbital solution of Io. 

Significant contributions of astrometry to Io's orbital solution occur for radio science true-to-formal-error ratios of 
4 and higher (for the along-track and normal direction). This shows that optical space-based astrometry can improve 
and/or validate the radio science solution. Reductions in the obtainable uncertainties for the COF-COM-offset range
from about 20 to 50 per cent - depending on the number of observations - using suitable algorithms to select the epochs 
at which observations are to be simulated. In particular, observations during the high-inclination phase have proven 
especially beneficial.

Our results show that constraints on the COM-COF offset of Io could be obtained from astrometry at the level 
100 m - 1 km, depending on the quantity and planning of the observations. This could provide a novel data point to 
constrain Io's interior. Moreover, the astrometric data will provide independent validation - and possibly 
improvement - of the orbital solution of Io. 

Download and Installation
------------

Software-architecture-wise, this work heavily relies on the numerical capabilities provided by the TU Delft 
Astrodynamics Toolbox (tudat). To install tudat, the reader is primarily directed to the excellent
[documentation](https://tudat-space.readthedocs.io) as well as the source code – including a detailed step-by-step 
installation guide – located in a dedicated [repository](https://github.com/tudat-team/tudat-bundle). Note that the
following versions have been used for the individual tudat-submodules:

- tudat (2.12.1.dev28)
- tudat-resources (2.2)
- tudatpy (0.7.3.dev20)

As of now, down- and upwards compatibility to other than the above-specified versions have not been explicitly tested 
and are thus not ensured – but might nevertheless exist...

To ensure full functionality, given their large size of several gigabytes, the reader is kindly asked to 
him/her/themselves download the required SPICE kernels for both the trajectories of 
[JUICE](https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/) and [Cassini](https://ftp.imcce.fr/pub/softwares/caviar/) 
also including all associated planetary ephemerides from the respective ftp-servers. While for JUICE we have used the
(simulated, pre-flight) baseline 150lb CReMA 5.1 kernels, the trajectory of Cassini is based on the officially 
distributed post-flight kernels. Please ensure that the kernel-files are placed in their proper directories
(juice_kernels and cassini_kernels, respectively) and state your local PATH_VALUE in the associated JUICE .txt file. 
On a final note, bear in mind that any kernels concerning Cassini are solely required in the context of the limb-fitting 
coefficient estimation (cf. Appendix A of the paper) and do not influence the general functionality of the remainder 
of the code.

Function and Purpose
--------------------

```
.
├── src
│   │   
│   ├── core
│   │   │   
│   │   ├── covariance_analysis
│   │   │   ├── estimation_plotting.py                     ### Figures 6, 7, 8, and 9
│   │   │   ├── formal_errors_evolution_plotting.py        ### Figure 10
│   │   │   ├── monte_carlo_analysis.py                    ### Sections 3.2.2 and 5.2
│   │   │   ├── observation_angles_plotting.py             ### Section 5.1 and Figures 4 and 5
│   │   │   └── quality_radio_science.py                   ### Section 5.3
│   │   │
│   │   ├── space_based_astrometry
│   │   │   └── space_based_astrometry_utilities.py        ### Sections 3.1 and 3.2.1
│   │   │
│   │   └── tudat_setup_util
│   │       │
│   │       ├── env
│   │       │   ├── env_full_setup.py                      ### Section 2.1
│   │       │   ├── env_utilities_galilean_moons.py        ### Section 2.3
│   │       │   └── env_utilities_jupiter.py               ### Section 2.3
│   │       │
│   │       ├── propagation
│   │       │   └── prop_full_setup.py                     ### Section 2.2
│   │       │
│   │       └── spice
│   │           └── juice_kernels                          ### Put your JUICE kernels here!
│   │       
│   └── misc
│       ├── cassini_kernels                                ### Put your Cassini kernels here!
│       ├── estimation_dynamical_model.py                  ### Section 4.3 and indirectly Figure 1
│       └── limb_fitting_uncertainty_estimation.py         ### Section 3.1.1 and Appendix A
│   
├── tests
│   ├── validation.py                                      ### Sections 2 and 3 – validation
│   └── validation_plotting.py                             ### Figure 3
│   
└── README.md
```

Authors and Acknowledgment
--------------------

Finally, I would like to express my eternal gratitude for the two best supervisors (and co-authors) I could have ever 
dreamt of - Dominic and Sam. Whilst one of the biggest advantages of having two supervisors is receiving twice the 
amount of feedback - this could simultaneously turn out to be a significant disadvantage. Yet, with all your input and 
ideas, as well as motivating feedback I never once regretted the decision of choosing the two of you as supervisors, 
and I am sure this work has greatly benefited from it. Even though it has also been a lot of hard work having to wrap 
my head around all those covariance matrices, you made this adventure so much fun.
