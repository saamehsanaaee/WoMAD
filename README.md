# Working Memory Architecture and Demand (WoMAD) Model

This reposiroty will house the WoMAD model project code which is an extension of our previous work during the NMA ISP, called [WMAD](https://github.com/saamehsanaaee/WMAD-Montbretia_Cabinet-ISP) with a micropublication on Zenodo, titled "[Parallel GNN-LSTM Model Predicting Working Memory Involvement during Language and Emotion Processing](https://doi.org/10.5281/zenodo.15126506)."

WoMAD is a modular neural network designed to analyze fMRI and BOLD time series data. The model uses spatiotemporal biosignals from task-based functional MRI datasets like the Human Connectome Project (HCP) to predict probability scores of node-based activity levels in the brain during tasks based on patterns of Working Memory (WM) activity in the brain.

WoMAD has two independent modules: Information flow and Core.

The information flow module follows the spatiotemporal activity of tasks in the data to creatae a 3D graph of the "moving activity" in the brain predicted in both unsupervised and supervised ways. The goal of the information flow module is to find the path of information through a brain as a task is being completed.

The core module is comprised of four submodules. First, the BOLD time series data is presented to a segmentation module to be labeled based on activity. The labeled output is then passed to two parallel submodules, a convolutional 4D network and an LSTM, which analyze the signals spatiotemporally and temporally, respectively. The final submodule is a fusion layer that combines ourputs from our parallel pair of submodules to produce two outputs: (a) A probability score of overall activity and (b) voxel-wise (or parcel-wise) probability scores of activity which will allow us to analyze the location of activity for the analyzed task and statistically predict the extent of WM involvement in a given task.

## License
This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

See the [LICENSE](LICENSE) file for more details.
