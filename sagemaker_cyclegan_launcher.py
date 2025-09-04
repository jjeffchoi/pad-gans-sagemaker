# sagemaker_cyclegan_launcher.py
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
from datetime import datetime

# --- Configuration ---
S3_BUCKET_NAME = "gans-training"
S3_DATA_PREFIX = "datasets/pad_cyclegan" # The S3 prefix where trainA/ and trainB/ are located
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
JOB_BASE_NAME = f"cyclegan-pad"
S3_OUTPUT_PREFIX = f"models/{JOB_BASE_NAME}"

# Get your AWS account's execution role
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role_name = "Your-SageMaker-Execution-Role-Name"  # <--- REPLACE with your role name
    role = iam.get_role(RoleName=role_name)['Role']['Arn']

# ===================================================================
# OPTION 1: LAUNCH A NEW TRAINING JOB
# ===================================================================

# --- Define Hyperparameters for the new job ---
hyperparameters = {
    'epochs': 50,
    'bs': 8,  # Batch size
    'lr': 0.0002,
    'lambda-cyc': 10.0,
    'samples-per-epoch': 2,    # New: Save samples twice per epoch
    'checkpoint-epochs': 5,  # New: Save a full checkpoint every 5 epochs
    'verbose': True
    # NOTE: Do not include the 'resume' argument for a new job
}

# --- Setup the SageMaker Estimator ---
cyclegan_estimator = PyTorch(
    entry_point='train.py',
    source_dir='./src',
    instance_type='ml.g4dn.2xlarge', # A slightly larger instance might be better for GANs
    base_job_name=JOB_BASE_NAME,
    instance_count=1,
    role=role,
    framework_version='2.0',
    py_version='py310',
    hyperparameters=hyperparameters,
    output_path=f's3://{S3_BUCKET_NAME}/{S3_OUTPUT_PREFIX}/',
    # Use SageMaker's managed checkpointing feature
    checkpoint_s3_uri=f's3://{S3_BUCKET_NAME}/{S3_OUTPUT_PREFIX}/checkpoints/',
    sagemaker_session=sagemaker.Session(),
    max_run=5 * 24 * 60 * 60 # 5 days
)

# --- Start the Training Job ---
s3_data_path = f"s3://{S3_BUCKET_NAME}/{S3_DATA_PREFIX}"
cyclegan_estimator.fit({'training': s3_data_path})

print("\n✅ New CycleGAN Training Job Launched!")
print("Track its progress in the AWS SageMaker console under 'Training jobs'.")


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
#     'verbose': True,
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
