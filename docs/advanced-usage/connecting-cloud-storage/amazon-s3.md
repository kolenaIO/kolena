---
icon: simple/amazons3
---

# :simple-amazons3: Amazon S3

Integrations can be established to Amazon S3.

To get started, ensure you have administrator access within Kolena.
Navigate to the "Integrations" tab on the [:kolena-organization-16: Organization Settings](https://app.kolena.io/redirect/organization?tab=integrations) page and click "Add Integration", then "Amazon S3".

### 1. Select Integration Scope

Amazon S3 Integrations load bucket objects by creating an IAM Role for Kolena to assume.

By default, Kolena will assume this Role when loading objects from any permitted bucket.
Alternatively, if you wish for Kolena to assume this Role only while loading objects from one bucket, uncheck
"Use role by default for all permitted buckets?" and specify the S3 bucket name.

Click "Next".

!!!note "Note: scoping Integrations"

    If "Use role by default for all permitted buckets?" is selected, Kolena will load any locators beginning with `s3://` by assuming the Role
    configured for this Integration and generating a presigned URL.
    Scoping the Integration to one bucket (e.g. `my-bucket`) means Kolena will only assume the Role when generating presigned URLs for locators of the form
    `s3://my-bucket/*`.

### 2. Create an Access Policy in AWS

If you selected "Use role by default for all permitted buckets?" in the previous step, you must now choose which buckets
Kolena is permitted to load objects from.
Enter these bucket names.

When you have entered the bucket names, you will see an "Access Policy JSON" rendered to your page.
Copy this JSON.

!!!note "Note: IAM Write Permission Required"

    You will require IAM write permissions within your AWS account to perform the next step.

In your AWS console, navigate to the <a target="_blank" href="https://console.aws.amazon.com/iamv2/home#/policies">IAM policies page</a>.
Click the "Create Policy" button and select the "JSON" tab.
Paste the "Access Policy JSON" copied previously.
Click through the "Next" buttons, adding the desired name, description, and tags.

### 3. Create a Role For Kolena to Assume

Return to Kolena and copy the "Trust Policy JSON".

In your AWS console, navigate to the <a target="_blank" href="https://console.aws.amazon.com/iamv2/home#/roles">IAM roles page</a>.
Click the "Create role" button and select 'Custom trust policy".
Paste the "Trust Policy JSON" you copied above and click "Next".
Search for and select the access policy created in [step 2](#2-create-an-access-policy-in-aws).
Provide a role name and review the permissions, then click "Create role".

**Copy the Role's ARN for use in the next step.**

### 4. Save Integration

Return to Kolena and fill in the remaining fields for the Integration and then click "Save".

| Field            | Description                                                                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Role ARN         | The ARN of the role created in [step 3](#3-create-a-role-for-kolena-to-assume)                                                                     |
| Endpoint URL     | The fully qualified endpoint of the webservice. This is only required when using a custom endpoint (for example, when using a local version of S3) |
| Region           | The region your buckets will be accessed from (e.g. `us-east-1`)                                                                                   |
| Force Path Style | Whether to force path style URLs for S3 objects (e.g., https://s3.amazonaws.com// instead of https://.s3.amazonaws.com/)                           |

### 4. Allow CORS Access to Bucket

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
