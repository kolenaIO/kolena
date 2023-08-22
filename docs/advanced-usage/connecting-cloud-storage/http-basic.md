Integrations can be established using HTTP Basic Auth.

To get started, ensure you have administrator access within Kolena.
Navigate to the "Integrations" tab on the [:kolena-organization-16: Organization Settings](https://app.kolena.io/redirect/organization?tab=integrations) page and click "Add Integration", then "HTTP Basic".

### 1. Save Integration on Kolena

On the [Integrations tab](https://app.kolena.io/redirect/organization?tab=integrations), fill in the fields for the integration and then click "Save".

| Field | Description |
|---|---|
| URL Origin | The origin of the domain you wish to load data from. Ensure you omit the protocol (e.g. `https://`) |
| Username | The username for your http basic auth system |
| Password | The password (optional) for your http basic auth system |

Any locators beginning with `https://<URL Origin>` will be loaded using this integration.
