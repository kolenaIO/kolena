# Google Cloud Storage integration

For full functionality, Kolena requires read access to the image data within a shared GCP bucket. 
 This is done by granting a Kolena service user read access to the bucket under your organization.

## 1) Create Service Account - Kolena ##
Kolena will create and send a service account for your Organization that will be used to read images for this bucket.
Please grant read access to the desired shared bucket to this account.

Example account name: `kolena-read-user-1234@kolena-prod.iam.gserviceaccount.com`

## 2) Grant Service Account read access ##
The screenshots below show steps for granting an `kolena-read-user-1234@kolena-prod.iam.gserviceaccount.com` access to
a `kolena_access_test` bucket.

- Open the bucket that contains your data
- Click the permissions menu
- Select the Grant Access button and grant the service account provided in step 1 the Storage Object viewer Role in the GCP console.

- The Storage Object Viewer role offers the following permissions:
  - Grants access to view objects and their metadata, excluding ACLs.
  - Can also list the objects in a bucket.

## 3) Provide CORS access for the bucket to Kolena ##
Create a json file example_cors_file.json with the following content.
(Please ensure that you update the origin with your Kolena platform URL)

```json
[
    {
      "origin": ["https://app.kolena.io/<your-organization>"],
      "method": ["GET"],
      "responseHeader": ["Content-Type"],
      "maxAgeSeconds": 3600
    }
]
```

Run the below command (requires `gsutil` to be installed) (Please ensure that you replace the filename and `example_bucket` with the correct information)

`gsutil cors set example_cors_file.json gs://example_bucket`