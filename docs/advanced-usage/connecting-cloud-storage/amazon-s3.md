---
icon: simple/amazons3
---

# :simple-amazons3: Amazon S3

Integrations can be established to Amazon S3.

To get started, ensure you have administrator access within Kolena.
Navigate to the "Integrations" tab on the [:kolena-organization-16: Organization Settings](https://app.kolena.io/redirect/organization?tab=integrations) page and click "Add Integration", then "Amazon S3".

### 1. Allow CORS Access to Bucket

CORS permissions are required for the Kolena domain to render content from your bucket.

Navigate to your S3 bucket inside your AWS console.
Click on the "Permissions" tab and navigate to the "Cross-origin resource sharing (CORS)" section.
Click "Edit" and add the following JSON snippet:

```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET"],
    "AllowedOrigins": ["https://app.kolena.io"],
    "ExposeHeaders": []
  }
]
```

### 2. Create an Access Policy in AWS

First ensure you have IAM write permissions within your AWS account.
In your AWS console, navigate to the IAM policies page.
Click the "Create Policy" button and select the "JSON" tab.

Copy and paste the following JSON policy:


!!!note "Note: Update Resource Name"

    After copying the JSON below, ensure that `share-with-kolena` is replaced with the appropriate bucket you wish to provide access to.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ListBucket",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::share-with-kolena",
        "arn:aws:s3:::share-with-kolena/*"
      ]
    }
  ]
}
```

Click through the "Next" buttons, adding the desired name, description, and tags.

### 3. Create a Role For Kolena to Assume

Return to the Kolena platform [Integrations tab](https://app.kolena.io/redirect/organization?tab=integrations).

On the "Create Amazon S3 Integration" page, click "Generate a Principal ARN".
This will create a Principal which will be referenced in a trust policy.
Setting an External Id to include within the trust policy is recommended.

Once these steps are complete, copy the JSON which appears on the page.
This will be of the form:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "KolenaAssumeRole",
      "Effect": "Allow",
      "Principal": {
        "AWS": "<< Principal ARN >>"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": ["<< External Id >>"]
        }
      }
    }
  ]
}
```

Navigate to the IAM roles page in your AWS console.
Click the "Create role" button and select 'Custom trust policy".
Paste the JSON you copied above and click "Next".
Search for and select the access policy created in [step 2](#2-create-an-access-policy-in-aws).
Provide a role name and review the permissions, then click "Create role".

Copy the role's ARN for use in the final step.

### 4. Save Integration

Return to the Kolena platform [Integrations tab](https://app.kolena.io/redirect/organization?tab=integrations).

By default, any locators beginning with `s3://` will be loaded using this integration.

!!!note "Note: scoping integrations"

    Optionally, each integration can be scoped to a specific bucket such that only locators of the pattern `s3://<specific-bucket>/*` will be loaded using the integration.
    This can be necessary if multiple integrations are required.
    Unchecking "Apply to all buckets by default?" and specifying a bucket will enable this behavior.

Fill in the fields for the integration and then click "Save".

| Field            | Description                                                                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Role ARN         | The ARN of the role created in [step 3](#3-create-a-role-for-kolena-to-assume)                                                                     |
| Endpoint URL     | The fully qualified endpoint of the webservice. This is only required when using a custom endpoint (for example, when using a local version of S3) |
| Region Name      | The region your buckets will be accessed from                                                                                                      |
| Force Path Style | Whether to force path style URLs for S3 objects (e.g., https://s3.amazonaws.com// instead of https://.s3.amazonaws.com/)                           |
