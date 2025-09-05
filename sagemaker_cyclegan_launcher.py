# sagemaker_cyclegan_launcher.py
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
from datetime import datetime
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

# --- Configuration ---
S3_BUCKET_NAME = "gans-training"
S3_DATA_PREFIX = "datasets/pad_cyclegan" # The S3 prefix where trainA/ and trainB/ are located
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
JOB_BASE_NAME = f"cyclegan-pad"
S3_OUTPUT_PREFIX = f"models/{JOB_BASE_NAME}"
S3_CHECKPOINT_PATH = f"s3://{S3_BUCKET_NAME}/checkpoints/{JOB_BASE_NAME}"

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role_name = "SageMaker-Role"
    role = iam.get_role(RoleName=role_name)['Role']['Arn']

# ===================================================================
# OPTION 1: LAUNCH A NEW TRAINING JOB
# ===================================================================

# --- STEP 1: Define Static Hyperparameters ---
# These are parameters that will be the same for ALL tuning jobs.
static_hyperparameters = {
    'epochs': 50,
    'bs': 8,
    'samples-per-epoch': 1,
    'checkpoint-epochs': 10
}

# --- Setup the Base PyTorch Estimator ---
# This defines the template for each training job the tuner will launch.
# Notice that 'lr' and 'lambda-cyc' are NOT defined here.
cyclegan_estimator = PyTorch(
    entry_point='train.py',
    source_dir='./src',
    instance_type='ml.g4dn.2xlarge',
    instance_count=1,
    role=role,
    framework_version='2.0',
    py_version='py310',
    hyperparameters=static_hyperparameters, # Use the static ones
    sagemaker_session=sagemaker.Session(),
    checkpoint_s3_uri=S3_CHECKPOINT_PATH,
    max_run=3 * 24 * 60 * 60 # 3 days max per job
)

# --- STEP 2: Define the Objective Metric ---
# We need to tell SageMaker how to find the performance metric in the training logs.
# Your script prints: "Epoch 50 | G Loss: 1.2345 | D Loss: 0.4567"
# This Regex will capture the floating point number right after "G Loss: ".
objective_metric_name = 'avg_generator_loss'
metric_definitions = [{
    'Name': objective_metric_name,
    'Regex': r"Avg G_loss: ([\d\.]+),"
}]

# --- STEP 3: Define the Hyperparameter Search Space ---
# This is where you specify the ranges for the parameters you want to tune.
hyperparameter_ranges = {
    'lr': ContinuousParameter(0.0001, 0.0005),  # Search for the best learning rate between 0.0001 and 0.0005
    'lambda-cyc': IntegerParameter(5, 20)      # Search for the best cycle consistency weight between 5 and 20
}

# --- STEP 4: Create the HyperparameterTuner Object ---
tuner = HyperparameterTuner(
    estimator=cyclegan_estimator,           # The base estimator to use
    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=metric_definitions,
    max_jobs=2,                            # The total number of training jobs to run
    max_parallel_jobs=1,                    # Run up to 2 jobs at the same time
    objective_type='Minimize'               # We want to find the job with the LOWEST generator loss
)

# --- Start the Hyperparameter Tuning Job ---
s3_data_path = f"s3://{S3_BUCKET_NAME}/{S3_DATA_PREFIX}"
tuner.fit({'training': s3_data_path})

print("\n✅ CycleGAN Hyperparameter Tuning Job Launched!")
print(f"Track its progress in the AWS SageMaker console under 'Hyperparameter tuning jobs' -> '{tuner.latest_tuning_job.name}'.")

# ===================================================================
# OPTION 2: RESUME A PREVIOUS TRAINING JOB (UNCOMMENT TO USE)
# ===================================================================
#
# # 1. Find the full name of the job you want to resume from the SageMaker console.
# previous_job_name = "cyclegan-pad-YYYY-MM-DD-HH-MM-SS-XXX" # <--- REPLACE
#
# # 2. Construct the S3 path where the previous job's outputs were saved.
# # This path contains the checkpoints, models, and samples.
# checkpoint_s3_path = f's3://{S3_BUCKET_NAME}/{S3_OUTPUT_PREFIX}/{previous_job_name}/output/'
#
# # 3. Define hyperparameters for the resumed job. You can increase epochs, change the learning rate, etc.
# hyperparameters_for_resume = {
#     'epochs': 100,  # Continue training for 50 more epochs
#     'lr': 0.0001,   # Optionally, decrease learning rate
#     'bs': 8,
#     'lambda-cyc': 10.0,
#     'samples-per-epoch': 2,
#     'checkpoint-epochs': 5,
#     # CRITICAL: Specify which checkpoint file to load from the previous job
#     'resume': 'checkpoint_final.pth' # Or e.g., 'checkpoint_step_25000.pth'
# }
#
# # 4. Create a new estimator, pointing it to the previous job's output as the checkpoint source.
# resume_estimator = PyTorch(
#     entry_point='train.py',
#     source_dir='./src',
#     instance_type='ml.g4dn.2xlarge',
#     base_job_name=f"resume-{JOB_BASE_NAME}",
#     instance_count=1,
#     role=role,
#     framework_version='2.0',
#     py_version='py310',
#     hyperparameters=hyperparameters_for_resume,
#     output_path=f's3://{S3_BUCKET_NAME}/{S3_OUTPUT_PREFIX}/',
#     # THIS IS THE KEY: Tell SageMaker where to get the checkpoint data from
#     checkpoint_s3_uri=checkpoint_s3_path,
#     sagemaker_session=sagemaker.Session()
# )
#
# # 5. Start the resumed training job.
# resume_estimator.fit({'training': s3_data_path})
#
# print(f"\n✅ Resuming training from job '{previous_job_name}'")
