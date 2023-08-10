---
subtitle: MinIO, Oracle
---

Integrations can be established to S3-compatible systems.
Supported systems include:
* [MinIO](https://min.io/docs)
* [Oracle Object Storage](https://docs.oracle.com/en-us/iaas/Content/Object/Tasks/s3compatibleapi.htm)

To get started, ensure you have admin access within Kolena.
Navigate to your [Organization Settings's Integration Tab](https://app.kolena.io/redirect/organization?tab=integrations) and click "Add Integration", then "MinIO".

Steps performed outside of Kolena are shown for a subset of possible S3-compatible systems.
You may need to consult documentation for your provider to perform equivalent steps.

### 1. Create a Service User for Kolena


=== "`MinIO`"

    ```shell
    mc admin user add <deployment_alias> <kolena_user> <secret_access_key>
    ```


### 2. Create an Access Policy

Create a policy to allow read access for a bucket or set of buckets.

Save the following JSON policy to a file called `/tmp/kolena-policy.json`, replacing `s3://share-with-kolena` with the appropriate bucket(s):


```json
{
    "Version": "2012-10-17",
    "Statement": [{
        "Sid": "S3ListBucket",
        "Effect": "Allow",
        "Action": [
            "s3:GetObject",
            "s3:ListBucket"
        ],
        "Resource": [
            "arn:aws:s3:::share-with-kolena",
            "arn:aws:s3:::share-with-kolena/*",
        ]
    }]
}
```

!!!note "Note: bucket names"

    Please note that bucket names must follow [S3 naming rules](https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html)

Next, create the policy and attach the policy to the service user created in [step 1](#1-create-a-service-user-for-kolena):

=== "`MinIO`"

    ```shell
    mc admin policy create <deployment_alias> kolenaread /tmp/kolena-policy.json
    mc admin policy attach <deployment_alias> kolenaread --user <kolena_user>
    ```

​   ​
### 3. Save Integration on Kolena

Return to the Kolena platform [Integration Tab](https://app.kolena.io/redirect/organization?tab=integrations)

By default, any locators beginning with `s3://` will be loaded using this integration.

!!!note "Note: scoping integrations"

    Optionally, each integration can be scoped to a specific bucket such that only locators of the pattern `s3://<specific-bucket>/*` will be loaded using the integration.
    This can be necessary if multiple integrations are required.
    Unchecking "Apply to all buckets by default?" and specifying a bucket will enable this behavior.

Fill in the fields for the integration and then click "Save".

| Field | Description |
|---|---|
| Access Key Id | The username (`<kolena_user>`) of the user created in [step 1](#1-create-a-service-user-for-kolena) |
| Secret Access Key | The secret key (`<secret_access_key>`) of the user created in [step 1](#1-create-a-service-user-for-kolena) |
| Endpoint | The hostname or IP address of your S3-compatabile service |
| Port | The optional port to access your S3-compatabile service |
| Region | The region your buckets will be accessed from |
