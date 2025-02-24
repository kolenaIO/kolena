---
icon: kolena/diagram-tree-16
---

# :kolena-diagram-tree-16: Setting Up SCIM on Kolena

This guide outlines the steps to configure SCIM on the Kolena platform.
It covers enabling SCIM, configuring groups for role mapping,
managing users, and troubleshooting common issues.

## Prerequisites

- Administrative access to your organization's Workspace.
- Administrative access to your Kolena organization settings.

## Step 1: Create and Configure Groups

Kolena automatically assigns user roles based on the names of the groups they belong to.
The group names must include specific keywords: `admin`, `member`, or `reader`.

### Create Groups in Workspace

In the Workspace admin console, create groups with names containing one of the following keywords:

- `admin` for administrative users
- `member` for standard users
- `reader` for read-only users

#### Examples

- `app-kolena-admin` (for admins)
- `app-kolena-member` (for standard users)
- `app-kolena-reader` (for read-only users)

### Add Users to Groups

Assign users to the appropriate groups based on their desired permission levels in Kolena.

!!! important
    - The keywords (`admin`, `member`, `reader`) are case-sensitive and must be part of the group name.
    - If renaming an existing group, you may need to remove the SCIM configuration, rename the group,
      and re-enable SCIM to ensure proper syncing.

## Step 2: Enable SCIM on Kolena

1. Sign in to your Kolena account with administrative privileges.

2. Navigate to the Organization Settings screen.
![Organization Settings Navigation](../assets/images/scim-setup-image.png)

### Configure SCIM

1. Copy integration link and open in a new tab.
2. Follow the prompts to connect Kolena to your Directory Sync Provider (e.g. Google Cloud Identity, Microsoft Entra, Okta).
   This typically involves generating an API key or OAuth token depending on your provider.
![Directory Provider Setup Screen](../assets/images/scim-active-directory.png)

### Verify Connection

After enabling SCIM, you should see a confirmation message indicating a successful connection and that groups are being synced.
If you encounter a timeout or error, ensure your group names are correctly set up (see Step 1).

!!! note "Important Notes"
    - Configure your groups with the correct naming convention before enabling SCIM to avoid connection issues.
    - Kolena uses a Directory Sync Provider to facilitate SCIM integration with Workspaces.

## Understanding User Synchronization

Once SCIM is enabled, user synchronization between Workspace and Kolena happens automatically.
However, there are some important aspects to understand about the synchronization process.

### Initial Sync

When you first enable SCIM with correctly named groups pre-populated with users, Kolena will sync all group members.
Ensure the SCIM connection is re-established if you rename groups.

### Ongoing Sync

Changes to group membership (e.g., adding or removing users) are synced automatically.
Kolena's sync interval with the Directory Sync Provider is approximately 3 minutes,
but Workspace may introduce additional delays (up to 15 minutes or more) before sending events to the provider.

!!! note
    If users don't appear in Kolena after adding them to a group, wait at least 15 minutes to account for sync frequency.
    Check group naming if the issue persists.

## User Permissions and Access Management

### Permission Levels

Kolena assigns permissions automatically based on group membership:

- `admin` groups: Users gain administrative privileges
- `member` groups: Users gain standard user privileges
- `reader` groups: Users gain read-only access

### Multiple Group Memberships

If a user is in multiple groups (e.g., `app-kolena-member` and `app-kolena-admin`),
they are granted the highest permission level (in this case, admin).

### Revoking Access

To revoke a user's access to Kolena, remove them from the relevant group(s) in Workspace.
This triggers a `GROUP_USER_REMOVED` event, which Kolena syncs to disable the user's account.
The change will reflect in Kolena after the sync delay (up to 15 minutes or more).

!!! warning
    Disabling a user's account might not immediately revoke their Kolena access, as this event may not be visible to the
    Directory Sync Provider. Always remove the user from the group to ensure proper deprovisioning.
