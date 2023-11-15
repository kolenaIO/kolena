---
icon: simple/cloudfront
---

# Connect Amazon CloudFront SDN to speed up S3 asset loading

Loading assets from Amazon S3 can be slow, especially when there are multiple assets that needs to be loaded at the same
time. Since S3 only supports http1.1, which only allows a single request and response to be sent over a single connection
at a time, the number of connections per domain allowed is capped by the browser, in most cases a maximum of 6 connections
can be made to S3 at the same time. It caused a lot of request to be queued. This can be addressed by setting up a
CloudFront distribution in front of S3 which supports http3 protocol, http3 protocol allows multiple requests and
responses to be sent over a single connection. In our experimentation, we have seen a 2x speed up in loading assets
from s3 with a cloudfront setup when loading 25 large images at the same time.

## Setup
To safeguard your private s3 assets, we recommend that you use Origin Access Control to restrict access. We would require
you to create a key pair and a key group in CloudFront, and share the private key with us. The key pair will be used to
sign the cloudfront URL. You will have full control of the keys, you can remove, rotate or disable the key pair from your
AWS console and Kolena platform at any time.

[Official AWS Doc For Serving Private Content over Cloudfront using signed Url](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/PrivateContent.html)
### Create a Key Group with a key pair
1. Follow the [Official AWS Doc for creating a key group with a key pair](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-trusted-signers.html#private-content-creating-cloudfront-key-pairs)
and follow the `Create a key pair for a trusted key group (recommended)` section.
2. Store the private key in a safe place, you will need to share it with us after setting up cloudfront.
### Create a CloudFront Distribution With Origin Access Control
1. Open AWS console and navigate to CloudFront service, click on "Create Distribution" button. Choose the S3 bucket that
  contains the asset you want to speed up as the "Origin Domain Name".
2. Select `Origin access control settings (recommended)` as the `Origin Access` and click on `Create Control Setting`.
3. Leave the rest of the Origin setting as default or modify to your need.
4. Select `HTTPS Only` as the `Viewer Protocol Policy`, and allow `HTTP GET and HEAD` as the `Allowed HTTP Methods`.
5. Select `Yes` in the `Restrict viewer access` section, and select `Trusted key groups (recommended)`, then add the key group you just created.
This gives all keys in that key group permission to generate signed URLs for the distribution.
6. Select both `HTTP/3` and `HTTP/2` as the `Supported HTTP Versions` section.
7. Leave the rest of the setting as default or modify to your need.
8. Click on `Create Distribution` button to create the distribution.
9. You should see a `The S3 bucket policy needs to be updated` banner on top of the page, click `Copy Policy` then
click `go to s3 bucket to update policy` on the banner
10. Update the bucket policy to allow the cloudfront distribution to read from the bucket. The policy to add should look like this
```json
{
  "Version": "2008-10-17",
  "Id": "PolicyForCloudFrontPrivateContent",
  "Statement": [
    {
      "Sid": "AllowCloudFrontServicePrincipal",
      "Effect": "Allow",
      "Principal": {
        "Service": "cloudfront.amazonaws.com"
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::your-datasets/*",
      "Condition": {
        "StringEquals": {
          "AWS:SourceArn": "arn:aws:cloudfront::1234567890:distribution/YourDistributionID"
        }
      }
    }
  ]
}
```

### Set up CloudFront on Kolena Platform
You could set up cloudfront on Kolena for a set of specific buckets in default s3 integration or for a specific bucket in
the bucket specific integration.
1. Navigate to the "Integrations" tab on the Organization Setting Page and click "Add Integration", then "Amazon S3".
2. Provide cloudfront private key and key pair id, and select the buckets you want to speed up with cloudfront.
3. More steps to be added once the frontend changes are made
