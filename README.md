# Int_mec_ML_KMC
 This repository demonstrates how the ML-KMC approach is applied to study interstitial-mediated diffusion in CSAs, from data preparation to performing diffusion simulations. For any questions or issues regarding the code, please feel free to contact me by biaoxu4@cityu.edu.hk; xubiao1189@hnu.edu.cn; xubiao1189@gmail.com.
# Purpose:
**This code is shared to help understand the methods in the research paper below or potentially apply the methods to explore the mechanism of interstitial sluggish diffusion in CSAs. 
***
## Research Title: Revealing the Interstitial-Mediated Sluggish Diffusion Mechanism in Concentrated Solid-Solution Alloys via Machine Learning-Integrated Kinetic Monte Carlo
***
### Research Group: Shijun ZHAO, J.J. Kai's Group in MNE, CityU
### Contact email: biaoxu4@cityu.edu.hk 
***
## Abstract:
![image](https://github.com/user-attachments/assets/15236417-3541-45a9-b3de-cf43121c7353)![image](https://github.com/user-attachments/assets/5ceb8d58-4cf8-4278-bf7b-e8e98c6c68b6)

Interstitial diffusion is a key process that influences phase stability and irradiation response in concentration solid solution alloys (CSAs) under non-equilibrium conditions. In this work, we developed an ML-KMC tool that integrates machine learning (ML) with kinetic Monte Carlo (KMC), achieving MD-level accuracy with the efficiency of KMC to study interstitial-mediated diffusion in CSAs. Using this tool, we identified that the interstitial-mediated sluggish diffusion occurs only when the reduction in the tracer correlation factor (ftr) outweighs the increase in jump frequency (ν). Unlike MD, the ML-KMC tool provides energy barrier information for both actual and potential migration paths during long-term diffusion, offering new insights into the underlying mechanisms of CSAs. Specifically, energy barrier differences between correlated migration patterns collaboratively form a 'route selector' that favors the migration of slower-diffusing components during dumbbell diffusion. This preference strengthens the correlation effect (decreasing ftr) and suppresses the increase in ν as the fast-diffusing component increases, resulting in interstitial-mediated sluggish diffusion. Furthermore, the current findings can be generalized to explain interstitial-mediated sluggish diffusion behavior in other CSA systems.
# Guide for using Code
***
**Description:**
***
**Running Platform:** All the code is developed based on MATLAB. So if you want to use the code for your own purpose, please make sure that you installed MATLAB already. (The platform of [Online Matlab](https://www.mathworks.com/products/matlab-online.html) or [Octave](https://octave.org/) may also be used to run this code well (not tested yet) ). In addition, the parallel skills are used in this code. We strongly recommend running the code on a machine that has multi-CPU cores or high-performance GPU modules, better to run them on the computer clusters (The reference shell scripts are also shared).  
***
**Code structure：** There are three folders: **ML-KMC, generate_data/NiFe73, train_ML_model**. They are used for different purposes. 
Due to the uploading limitation of GitHub, we share these files with relative data on
***
**"ML-kMC:"** This folder is for showing how the ML-KMC method is applied to perform a diffusion simulation. 
![image](https://github.com/user-attachments/assets/aaeceb08-5c51-431d-b16e-b47d75c065c4)

Main Function Files: The repository includes two main .m files: NiFe_kmc_stable.m and NiFe_kmc_stable_paralle.m. The former is used for a single CPU run, while the latter is designed for parallel execution on multiple CPUs.

Trained Model: The file optimal_model.mat contains the parameters of the well-trained model.

Supporting Function Modules: The function folder includes supporting function modules that handle data processing and provide supplementary functionality to the main program.

Load Data: The load_data folder contains per55, which provides a crystal structure identical to that used in ML training. The same function module is used to determine the reference coordinate system, ensuring it matches the one from the ML training process.
The main idea of this MATLAB code is to simulate the diffusion of interstitial structures (e.g., dumbbell interstitial atoms) within a metal lattice using a machine learning model and kinetic Monte Carlo (kMC) simulations. It aims to study the migration behavior and diffusion coefficients of interstitial atoms under different temperatures and chemical compositions. Below is an overview of the key workflow of the code:

Initialization:

Load the machine learning model (optimal_model) and import initial data (load_Data/per55), including initializing the lattice constant and atomic coordinates.
Creating Interstitial Structures:

Select a position to insert an interstitial atom and create an interstitial structure.
Call the creat_dumbell_interstitial function to generate a dumbbell interstitial atom at the selected position.
Coordinate Transformation and Nearest Neighbor Calculation:

Use the coord_transform function to transform the coordinates of the interstitial structure.
Use the update_interstitial_nn_kmc function to calculate the nearest neighbors of the dumbbell structure, which are then used to predict migration paths.
Migration Energy Prediction:

Predict the migration energy of each possible migration path using the machine learning model (ML_model).
Calculate the migration probabilities for each path and randomly choose the migration direction using the Monte Carlo method.
kMC Loop:

Update the coordinates of the interstitial atom at each step.
Calculate the migration step length and update the total simulation time using the Monte Carlo method.
Record the trajectory and Mean Squared Displacement (MSD) at each time step.
Storing and Outputting Results:

Store the simulation trajectories, migration paths, and related data in cell arrays for each ratio of elements.
Finally, perform a linear fit of MSD vs. time to determine the diffusion coefficient (D).
Plotting and Saving Results:

For each different ratio of elements, fit the MSD vs. time and plot the diffusion coefficients as a function of temperature.
Overall, the main purpose of this code is to investigate the migration behavior of interstitial atoms using a combination of a machine learning model and kinetic Monte Carlo simulations. The computed diffusion coefficient helps in understanding the atomic diffusion mechanism in metallic systems.


**generate_data:"** This folder provides a full example with NiFe73. The main file is a shell file mig_73_sub
![image](https://github.com/user-attachments/assets/85a8f71e-be90-4274-b3ea-e157f8dce19e)
***
Key notes:
The script performs multi-step calculations to determine migration energies of interstitial atoms in an Fe-Ni alloy.
It uses LAMMPS to calculate initial, intermediate, and final energies, allowing for the calculation of energy barriers.
The Nudged Elastic Band (NEB) method is used to calculate the transition energy barrier between two states.
The script leverages parallel computing (mpirun) to speed up the simulations, running them across multiple CPUs.
It repeatedly adjusts atom positions and configurations to obtain comprehensive data on possible interstitial migration paths and their associated energy barriers.
This script is primarily intended for atomistic-level simulation of defect migration in binary alloys, specifically designed to model the energy landscape of interstitial-mediated diffusion. The use of multiple LAMMPS runs for different configurations ensures the robustness of the energy calculations and the accuracy of the resulting migration barriers.

Other Input Files: There are four input files—inp1, inp2, inp3, and inp4. The NEB file is included in inp4. All these files are written in the LAMMPS format.

**"train_ML_model:"** This folder contains the code for training the ML model. The database for ML training was obtained through LAMMPS based on the interatomic potential of the embedded-atom method (EAM) type developed by Bonny et al. This potential has been widely applied to simulate the defect properties in Ni-containing CSAs, and it can reproduce reasonable and consistent results with density functional theory (DFT) calculations 19,24,37. The initial interstitial dumbbell configuration was created by inserting an atom to form a [100] dumbbell with a lattice atom located in the center of a 10×10×10 face-centered cubic (FCC) lattice. Different compositions of FexNi1-x were considered. Note that [100] dumbbells are the most stable interstitial form in FCC Ni. The migration of the [100] dumbbell to a [001] dumbbell through rotation in the {100}  plane was then simulated through the NEB method.   Due to the uploading size limitation of GitHub, we share the latest version with the Google Drive sharing [**train_ML_model**](https://drive.google.com/drive/folders/1RKA-LYCH1KthIbe68eprZEvXe0mOaoxS?usp=sharing)

The input of the ML model was the local atomic environment(LAE), described by the elemental types in the NN shells around the dumbbell interstitial, as the “INPUT” part shown in the Abstract figure. In this work, the LAE contains atoms within the 1st NN to the 10th NN based on our test. This LAE range is significantly larger than in vacancy cases, consistent with the greater spatial influence of interstitial dumbbells. The output labels were E_b, as shown by the “OUTPUT” part in the Abstract figure. 

![image](https://github.com/Jeremy1189/interstitial-diffusion/assets/85468234/4ab93fc3-c23e-48d9-a3a3-b19d1e44cd37)
