---
icon: kolena/globe-network-16
---

# :kolena-globe-network-20: HTTP Basic

Integrations can be established using HTTP Basic Auth.

To get started, ensure you have admin access within Kolena.
Navigate to your [Organization Settings's Integrations Tab](https://app.kolena.io/redirect/organization?tab=integrations) and click "Add Integration", then "HTTP Basic".

### 1. Save Integration on Kolena

On the [Integrations Tab](https://app.kolena.io/redirect/organization?tab=integrations), fill in the fields for the integration and then click "Save".

| Field | Description |
|---|---|
| URL Origin | The origin of the domain you wish to load data from. Ensure you omit the protocol (e.g. `https://`) |
| Username | The username for your http basic auth system |
| Password | The password (optional) for your http basic auth system |

Any locators beginning with `https://<URL Origin>` will be loaded using this integration.
