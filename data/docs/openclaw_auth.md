# OpenClaw Authentication

## Authentication Overview

### Summary
OpenClaw requires a valid authentication token to authorize agent requests.

### Token Types
- Static tokens
- OAuth tokens
- Provider-issued access tokens

---

## Error: token_missing

### Summary
The request failed because no authentication token was provided.

### Cause
The client did not include a valid token in the request headers or connection metadata.

### Resolution
1. Generate or retrieve a valid token.
2. Configure the token in the Control UI or agent settings.
3. Retry the request.

### Impact
The connection is rejected before processing.

---

## Error: unauthorized

### Summary
The provided token is invalid or expired.

### Cause
The token may be malformed, revoked, or expired.

### Resolution
1. Regenerate the token.
2. Verify correct provider configuration.
3. Update agent settings with the new token.

### Notes
Unauthorized requests are rejected immediately.
