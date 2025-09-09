# WoMAD Project Logs

This page includes the logs for our project. You'll be able to get a glance at the process (and hopefully, progress) of WoMAD when going through these logs.

The entries will be chronological and organized by date and version. We'll try our best to provide a clear history of the changes and tasks. You'll also find out signs of our junior-researcher-ness when you examine our solutions to issues we face. We'll laugh at those together.

---

## WoMAD v0.1.0

### Log 2025-09-07: First commits :) - v0.1.1

#### Deploy Docs Workflow (creation and errors):

Although the code is basically thin air at this point, Saameh decided it would be a good idea to *finalize* the structure of the repository. So, we got our first error. Shocking! I know!

This lead to creation of v0.1.1, a mostly empty repository that at least has a decent, mostly empty documents page on a Github page.

### Log 2025-09-09: Expansion of Documentation - v0.1.2

#### Expansion of documentation structure

Do you see a pattern? Before outlining the actual `.py` files, *someone* decided it would be a good idea to make the document and the Github page for our lovely WoMAD Model perfect. What is perfect? Nobody knows.

Our next tasks are:

* Outlining all `.py` files and complete their description docstrings
* Outlining `the-code.md` and `the-science.md` files
* Expand main `README.md` file

---

## Upcoming versions, logs, stages:

### WoMAD v0.2.0

* **Data Acquisition & Preprocessing**
* **Tasks:**
    * Develop functions to download the data
    * Parse the dataset and isolate trials using the EV files
    * Normalize data and save to a DataFrame

### WoMAD v0.3.0

* **Core Model Development**
* **Tasks:**
    * Define PyTorch `Dataset` and `DataLoader`
    * Create the 3D U-Net model (segmentation module)
    * Create a training loop (K-fold cross-validation)

### WoMAD v0.4.0

* **Parallel Modules & Fusion Module**
* **Tasks:**
    * Build the feature extraction modules:
        * Temporal module
        * Spatiotemporal module
        * GNN
    * Build the final fusion layer that combines their outputs

### WoMAD v0.5.0

* **Complete Model Evaluation & Interpretation**
* **Tasks:**
    * Evaluate model performance
    * Save model predictions as NIfTI files
    * Save (evaluation and prediction) plots

### WoMAD v1.0.0

* **First "release" of the "final" version of WoMAD**
* **Tasks:**
    * Finalize all code (well ... as final as it can be)
    * Finalize documentation
    * Add findings to `The Science` page
