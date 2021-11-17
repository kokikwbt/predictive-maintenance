# NASA Trubofun Jet Engine Dataset

The dataset summary is as follows.

| Dataset | Train trajectories | Test trajectries | Conditions | Fault Modes |
| :--- | ---: | ---: | ---: | ---: |
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 249* | 248* | 6 | 2 |

\* In official data description, \# of trian and test tarajectories in FD004 are 248 and 249 respectively. 
In fact, each of them are 249 and 248 in actual dataset.

## Experimental Scenario

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine � i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:

```
1)	unit number
2)	time, in cycles
3)	operational setting 1
4)	operational setting 2
5)	operational setting 3
6)	sensor measurement  1
7)	sensor measurement  2
...
26)	sensor measurement  26
```

### Sensor details  
Each sensor mesurements are below:  
1. T2 Total temperature at fan inlet °R  
2. T24 Total temperature at LPC outlet °R  
3. T30 Total temperature at HPC outlet °R    
4. T50 Total temperature at LPT outlet °R  
5. P2 Pressure at fan inlet psia  
6. P15 Total pressure in bypass-duct psia  
7. P30 Total pressure at HPC outlet psia  
8. Nf Physical fan speed rpm
9. Nc Physical core speed rpm  
10. epr Engine pressure ratio (P50/P2) --  
11. Ps30 Static pressure at HPC  
12. outlet psia phi Ratio of fuel flow to Ps30 pps/psi  
13. NRf Corrected fan speed rpm  
14. NRc Corrected core speed rpm  
15. BPR Bypass Ratio --  
16. farB Burner fuel-air ratio --
17. htBleed Bleed Enthalpy --
18. Nf_dmd Demanded fan speed rpm
19. PCNfR_dmd Demanded corrected fan speed rpm
20. W31 HPT coolant bleed lbm/s
21. W32 LPT coolant bleed lbm/s

## Reference
1. A. Saxena, K. Goebel, D. Simon, and N. Eklund,
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation",
in the Proceedings of the Ist International Conference on Prognostics
and Health Management (PHM08), Denver CO, Oct 2008.

## Reference for analysis  
1. Heimes, Felix. (2008). Recurrent neural networks for remaining useful life estimation. 1 - 6. 10.1109/PHM.2008.4711422. 
https://www.researchgate.net/publication/224358896_Recurrent_neural_networks_for_remaining_useful_life_estimation
2. Lee, Jay. (2008). A Similarity-Based Prognostics Approach for Remaining Useful Life Estimation of Engineered Systems. 
https://www.researchgate.net/publication/269167324_A_Similarity-Based_Prognostics_Approach_for_Remaining_Useful_Life_Estimation_of_Engineered_Systems
3. Malhotra, Pankaj & Tv, Vishnu & Ramakrishnan, Anusha & Anand, Gaurangi & Vig, Lovekesh & Agarwal, Puneet & Shroff, Gautam. (2016). Multi-Sensor Prognostics using an Unsupervised Health Index based on LSTM Encoder-Decoder. 1st SIGKDD Workshop on Machine Learning for Prognostics and Health Management. 
https://www.researchgate.net/publication/306376888_Multi-Sensor_Prognostics_using_an_Unsupervised_Health_Index_based_on_LSTM_Encoder-Decoder
4. https://hal.archives-ouvertes.fr/hal-01324729/document