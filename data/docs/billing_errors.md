# Billing Errors – Reference

## Error 1008 – Insufficient Balance

### Summary
Error 1008 indicates that the account does not have sufficient balance to process the request.

### Cause
The API request requires billing credits, but the account balance is zero or below the required amount.

### Resolution
1. Add funds to the account billing balance.
2. Verify billing status in the dashboard.
3. Retry the request after balance is updated.

### Impact
All billable API requests will fail until balance is replenished.

### Notes
This error is non-recoverable without user action.
