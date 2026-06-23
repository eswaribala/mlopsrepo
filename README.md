# Azure ML Train and Deploy

Python project for training and deploying a scikit-learn model with Azure Machine Learning.

## Install

```powershell
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and provide your Azure subscription, resource group,
Azure ML workspace, compute, and endpoint settings before running Azure ML workflows.

Steps
az extension add -n ml
az extension update -n ml
az ml -h
az ml workspace list --output table
if empty
az group create --name training_group --location eastus
az ml workspace create --name training-ws   --resource-group training_group --location eastus
#create cluster - cpu-cluster
az ml compute create --name cpu-cluster --type amlcompute --resource-group training_group --workspace-name training-ws
#set defaults
az configure --defaults group=training_group workspace=training-ws
#create job
az ml job create --file job.yml 
#access ml jobs using https://ml.azure.com/

az ml compute show --name cpu-cluster  --resource-group training_group --workspace-name training-ws --output table

#create model
az ml model create --name loan-approval-model --version 1 --type custom_model 
  --path azureml://jobs/mighty_spade_m8vdp8ns6w/outputs/artifacts/paths/outputs/loan_model.pkl --resource-group training_group
  --workspace-name training-ws

#verify the model
az ml model list --output table

# download the model
az ml model download --name loan-approval-model --version 1  --resource-group training_group --workspace-name training-ws  --download-path downloaded_model

#register providers
az account show --output table
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.ContainerRegistry

az provider register --namespace Microsoft.ContainerInstance

az provider register --namespace Microsoft.Insights
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.Network
az provider register --namespace Microsoft.ManagedIdentity
az provider register --namespace Microsoft.Authorization
az provider register --namespace Microsoft.Compute
az provider register --namespace Microsoft.ContainerService
az provider register --namespace Microsoft.Cdn
#check
az provider list  --query "[?namespace=='Microsoft.MachineLearningServices' || namespace=='Microsoft.Network' || namespace=='Microsoft.ContainerRegistry' || namespace=='Microsoft.Storage' || namespace=='Microsoft.KeyVault' || namespace=='Microsoft.Insights' || namespace=='Microsoft.ManagedIdentity' || namespace=='Microsoft.Authorization'].{Namespace:namespace, State:registrationState}" --output table

#run scope.py
az ml online-endpoint create --file endpoint.yml --resource-group training_group --workspace-name training-ws





#github azure credentials
az ad sp create-for-rbac  --name github-actions-sp --role Contributor 
  --scopes /subscriptions/7d267e35-6fb2-4a8d-b9ce-c127545512c8 
  --sdk-auth

  