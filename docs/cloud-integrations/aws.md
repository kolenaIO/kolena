# Amazon S3 integration

For full functionality, Kolena requires read access to the image data within a shared S3 bucket.
This is accomplished by creating a role for Kolena to assume.


## 1) Creating an Access Policy ##

These instructions assume that the data is stored in an `s3:://share-with-kolena` bucket.

- As a user with IAM write permissions go to the IAM policies page
- Click the `Create policy` button
- Select the `JSON` tab
- Paste the following JSON policy, making sure to update the bucket name accordingly:
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
                "arn:aws:s3:::share-with-kolena/*",
                "arn:aws:s3:::kolena-public-datasets",
                "arn:aws:s3:::kolena-public-datasets/*"
            ]
        }
    ]
}
```
- Click through the `Next` buttons, adding adding the desired tags, name, and description.
- In this example, we will name our policy `KolenaS3ReadAccess`.


## 2) Creating a role for Kolena to assume ##
- Please coordinate with the Kolena team for an AWS principal ARN
- Navigate to the AWS IAM roles page
- Click on the `Create role` button
- Configre the trust policy
  - Select `Custom trust policy` type
  - Paste the following JSON, making sure to use the ARN that was provided to you.
```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "KolenaAssumeRole",
			"Effect": "Allow",
			"Principal": {
			    "AWS": "PASTE KOLENA PRINCIPAL ARN HERE"
			},
			"Action": "sts:AssumeRole",
			"Condition": {
				"StringEquals": {
					"sts:ExternalId": [
						"example-external-id"
					]
				}
			}
		}
	]
}
```
- Please note that the `Condition` block is optional, and only if you wish to provide an `ExternalId` match string to Kolena.
- Click `Next`
- Search for and select the permissions policy created in step 1
  - Click 'Next'
- Provide a role name and review the permissions
  - Click `Create role` when ready


## 3) Allow CORS access from shared bucket ##
- In the `Permissions` section on the AWS S3 web console page for the bucket, add the following JSON snippet:
```json
[
    {
        "AllowedHeaders": [
            "*"
        ],
        "AllowedMethods": [
            "GET"
        ],
        "AllowedOrigins": [
            "https://app.kolena.io"
        ],
        "ExposeHeaders": []
    }
]
```
