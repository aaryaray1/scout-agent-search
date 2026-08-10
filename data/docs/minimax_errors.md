# MiniMax API Errors

## Error 1001 - Invalid Request Format

### Summary
Error 1001 occurs when the request payload does not match the expected API schema.

### Cause
Missing required fields or invalid JSON formatting.

### Resolution
1. Validate request structure.
2. Ensure required fields are present.
3. Retry with corrected payload.

### Affected APIs
- chat.completions
- text.generate

---

## Error 1008 - Insufficient Balance

### Summary
Error 1008 indicates insufficient billing balance to process the request.

### Cause
The account balance is too low to execute the requested API call.

### Resolution
1. Add funds to the billing account.
2. Confirm billing status.
3. Retry the request.

### Notes
This error blocks all billable endpoints until resolved.

---

## Error 1020 - Rate Limit Exceeded

### Summary
Error 1020 occurs when request volume exceeds the allowed rate limit.

### Cause
Too many requests sent in a short time window.

### Resolution
1. Reduce request frequency.
2. Implement exponential backoff.
3. Review rate limit quotas.

### Impact
Requests may succeed after cooldown period.
