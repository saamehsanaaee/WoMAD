# Working Memory Architecture and Demand (WoMAD) Model
This reposiroty will the WoMAD model project code which is an extension of our previous work during the NMA ISP called [WMAD](https://github.com/saamehsanaaee/WMAD-Montbretia_Cabinet-ISP) which is also published as a micropublication on Zenodo, titled "[Parallel GNN-LSTM Model Predicting Working Memory Involvement during Language and Emotion Processing](https://doi.org/10.5281/zenodo.15126506)."

---

## Requirements
All required packages have been listed in the `requirements.txt` file of the repository. To install all dependecies at once, please paste the following command in your terminal after navigating to the main directory:
```bash
pip install -r ./requirements.txt
```
After installing all requirements, please use the `aws config` command to add your access keys and then continue with your next steps. There is a page dedicated to this process in the HCP public pages: [How to Get Access to the HCP OpenAccess Amazon S3 Bucket](https://wiki.humanconnectome.org/docs/How%20to%20Get%20Access%20to%20the%20HCP%20OpenAccess%20Amazon%20S3%20Bucket.html#how-to-get-access-to-the-hcp-openaccess-amazon-s3-bucket)

When `awscli` is configured, you can start exploring the S3 Bucket with commands that are outlined in the [AWS CLI Command Reference](https://docs.aws.amazon.com/cli/latest/).

However, we decided to use Python packages to allow us to stream S3 buckets into our Python environment for easier analysis. These packages are also listed in the `./requirements.txt` file.

---

## Exploratory Data Analysis

---
